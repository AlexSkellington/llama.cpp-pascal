#include "server-schema.h"

#include "json-schema-to-grammar.h"

#include <unordered_set>
#include <deque>
#include <list>
#include <map>
#include <set>

namespace server_schema {

//
// llama.cpp-specific completion schema
//

// ---- local patch 0053 (engine parity): vLLM structured outputs -> llama grammar ----
static std::string vllm_gbnf_literal(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\x%02x", c); out += b; } else { out += (char) c; }
        }
    }
    return out + "\"";
}

// The schema converter wraps a pattern rule as root ::= "\"" (R) "\"" [space]; unwrap it into a raw rule.
// Unsupported patterns fall back to a generic string rule there: refused here instead of loosening silently.
// \d \w \s (and \D \W \S) -> explicit classes: llama.cpp's regex converter refuses the shorthands
static std::string vllm_expand_shorthand(const std::string & in) {
    std::string out;
    bool in_cls = false;
    for (size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '\\' && i + 1 < in.size()) {
            const char n = in[i + 1];
            const char lower = (char) tolower((unsigned char) n);
            const char * body = lower == 'd' ? "0-9" : lower == 'w' ? "a-zA-Z0-9_" : lower == 's' ? " \\t\\n\\r\\f\\v" : nullptr;
            if (body != nullptr) {
                const bool neg = n != lower;
                if (in_cls) {
                    if (neg) { throw std::runtime_error("structured_outputs.regex: negated shorthand inside a character class is not supported"); }
                    out += body;
                } else {
                    out += std::string(neg ? "[^" : "[") + body + "]";
                }
                ++i;
                continue;
            }
            out += c;
            out += n;
            ++i;
            continue;
        }
        if (c == '[' && !in_cls) { in_cls = true; } else if (c == ']' && in_cls) { in_cls = false; }
        out += c;
    }
    return out;
}

static std::string vllm_regex_grammar(const std::string & pattern) {
    std::string p = vllm_expand_shorthand(pattern);
    if (p.empty() || p[0] != '^') { p = "^" + p; }
    if (p.back() != '$') { p += "$"; }
    const std::string g = json_schema_to_grammar(json{{"type", "string"}, {"pattern", p}});
    std::string out;
    bool ok = false;
    size_t start = 0;
    while (start < g.size()) {
        size_t end = g.find('\n', start);
        if (end == std::string::npos) { end = g.size(); }
        std::string line = g.substr(start, end - start);
        const std::string pre = "root ::= \"\\\"\" (";
        if (line.rfind(pre, 0) == 0) {
            size_t close = line.rfind(") \"\\\"\"");
            if (close == std::string::npos || close < pre.size()) {
                throw std::runtime_error("structured_outputs.regex: unexpected converter output");
            }
            line = "root ::= " + line.substr(pre.size(), close - pre.size());
            ok = true;
        }
        out += line + "\n";
        start = end + 1;
    }
    if (!ok) {
        throw std::runtime_error("structured_outputs.regex: pattern not supported by llama.cpp's regex converter");
    }
    return out;
}

static void vllm_no_additional(json & s) {
    if (s.is_object()) {
        if (json_value(s, "type", std::string()) == "object" && !s.contains("additionalProperties")) { s["additionalProperties"] = false; }
        for (auto kv : s.items()) { vllm_no_additional(kv.value()); }
    } else if (s.is_array()) {
        for (auto & v : s) { vllm_no_additional(v); }
    }
}

// whitespace variants are user grammars: no chat-template prefill (vLLM does none), which a no-space grammar refuses
static bool vllm_ws_custom(const json & so) {
    return json_value(so, "disable_any_whitespace", false) || !json_value(so, "whitespace_pattern", std::string()).empty();
}

static std::string vllm_json_grammar(json schema, const json & so) {
    if (json_value(so, "disable_additional_properties", false)) { vllm_no_additional(schema); }
    std::string g = json_schema_to_grammar(schema);
    const bool no_ws = json_value(so, "disable_any_whitespace", false);
    const std::string wsp = json_value(so, "whitespace_pattern", std::string());
    if (!no_ws && wsp.empty()) { return g; }
    std::string out, space_rules;
    if (no_ws) {
        space_rules = "space ::= \"\"\n";
    } else {
        std::string r = vllm_regex_grammar(wsp);
        size_t pos;
        while ((pos = r.find("root")) != std::string::npos) { r.replace(pos, 4, "space"); }
        space_rules = r;
    }
    size_t start = 0;
    while (start < g.size()) {
        size_t end = g.find('\n', start);
        if (end == std::string::npos) { end = g.size(); }
        const std::string line = g.substr(start, end - start);
        if (line.rfind("space ::=", 0) != 0) { out += line + "\n"; }
        start = end + 1;
    }
    return out + space_rules;
}

static void vllm_set_grammar(task_params & params, common_grammar_type type, std::string g, const char * what) {
    if (!common_grammar_value(params.sampling.grammar).empty()) {
        throw std::runtime_error(std::string("Conflicting controls: ") + what + " and another grammar/json_schema");
    }
    params.sampling.grammar = {type, std::move(g)};
}

// ---- local patch 0055 (engine parity): vLLM structural_tag -> llama grammar ----
// xgrammar structural-tag formats compiled to GBNF. Free text is a character automaton over the trigger words
// (Aho-Corasick): text between tags never contains a trigger, a completed trigger always continues into a tag, excluded
// strings can never complete.
static std::u32string vllm_u32(const std::string & s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        uint32_t cp;
        int n;
        if (c < 0x80)              { cp = c;        n = 1; }
        else if ((c >> 5) == 0x6)  { cp = c & 0x1f; n = 2; }
        else if ((c >> 4) == 0xe)  { cp = c & 0x0f; n = 3; }
        else if ((c >> 3) == 0x1e) { cp = c & 0x07; n = 4; }
        else { throw std::runtime_error("structural_tag: invalid UTF-8 in a trigger/end/exclude string"); }
        if (i + n > s.size()) { throw std::runtime_error("structural_tag: truncated UTF-8 in a trigger/end/exclude string"); }
        for (int k = 1; k < n; ++k) { cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3f); }
        out.push_back(cp);
        i += n;
    }
    return out;
}

static std::string vllm_cp_esc(uint32_t c) {
    char buf[16];
    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) { buf[0] = (char) c; buf[1] = 0; }
    else if (c < 0x100)   { snprintf(buf, sizeof(buf), "\\x%02X", (unsigned) c); }
    else if (c < 0x10000) { snprintf(buf, sizeof(buf), "\\u%04X", (unsigned) c); }
    else                  { snprintf(buf, sizeof(buf), "\\U%08X", (unsigned) c); }
    return buf;
}

static std::string vllm_join(const std::vector<std::string> & v, const char * sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) { out += (i ? sep : "") + v[i]; }
    return out;
}

static bool vllm_gbnf_word(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-';
}

struct vllm_st_compiler {
    const common_grammar_builder & b;
    std::list<common_chat_schema_document> docs;
    int n = 0;

    explicit vllm_st_compiler(const common_grammar_builder & b) : b(b) {}

    std::string fresh(const char * base) { return "st" + std::to_string(n++) + "-" + base; }

    std::string rule(const std::string & name, const std::string & body) {
        const std::string got = b.add_rule(name, body);
        if (got != name) { throw std::runtime_error("structural_tag: grammar rule name clash on " + name); }
        return got;
    }

    static std::string lit(const std::string & s) { return s.empty() ? std::string("\"\"") : gbnf_format_literal(s); }
    static std::string lit_cp(uint32_t c) { return "\"" + vllm_cp_esc(c) + "\""; }

    std::string any_text() {
        const std::string nm = fresh("any");
        return rule(nm, "| . " + nm);
    }

    // text with no occurrence of words[i] except as its last characters, after which cont[i] follows ("" = this rule
    // ends there); excluded words never complete; end_ok lets the text stop anywhere
    std::string scan(const std::vector<std::u32string> & words, const std::vector<std::string> & cont,
                     const std::vector<bool> & excluded, bool end_ok) {
        struct node { std::map<uint32_t, int> next; int fail = 0; int out = -1; };
        std::vector<node> t(1);
        std::set<uint32_t> sigma;
        for (size_t w = 0; w < words.size(); ++w) {
            if (words[w].empty()) { throw std::runtime_error("structural_tag: empty trigger/end/exclude string"); }
            int u = 0;
            for (uint32_t c : words[w]) {
                sigma.insert(c);
                auto it = t[u].next.find(c);
                if (it == t[u].next.end()) {
                    t.emplace_back();
                    const int v = (int) t.size() - 1;
                    t[u].next[c] = v;
                    u = v;
                } else {
                    u = it->second;
                }
            }
            if (t[u].out < 0) { t[u].out = (int) w; }
        }
        std::deque<int> q;
        for (auto & kv : t[0].next) { q.push_back(kv.second); }
        while (!q.empty()) {
            const int u = q.front();
            q.pop_front();
            if (t[u].out < 0 && t[t[u].fail].out >= 0) { t[u].out = t[t[u].fail].out; }
            for (auto & kv : t[u].next) {
                const int v = kv.second;
                if (u != 0) {
                    int f = t[u].fail;
                    while (f != 0 && !t[f].next.count(kv.first)) { f = t[f].fail; }
                    auto it = t[f].next.find(kv.first);
                    t[v].fail = (it != t[f].next.end() && it->second != v) ? it->second : 0;
                }
                q.push_back(v);
            }
        }
        auto delta = [&](int u, uint32_t c) {
            while (u != 0 && !t[u].next.count(c)) { u = t[u].fail; }
            auto it = t[u].next.find(c);
            return it == t[u].next.end() ? 0 : it->second;
        };
        const std::string base = fresh("text");
        auto nm = [&](int u) { return base + "-" + std::to_string(u); };
        std::string other = "[^";
        for (uint32_t c : sigma) { other += vllm_cp_esc(c); }
        other += "]";
        if (sigma.empty()) { other = "."; }
        for (int u = 0; u < (int) t.size(); ++u) {
            if (t[u].out >= 0) { continue; }
            std::vector<std::string> alts;
            for (uint32_t c : sigma) {
                const int v = delta(u, c);
                const int w = t[v].out;
                if (w >= 0) {
                    if (excluded[w]) { continue; }
                    alts.push_back(lit_cp(c) + (cont[w].empty() ? std::string() : " " + cont[w]));
                } else {
                    alts.push_back(lit_cp(c) + " " + nm(v));
                }
            }
            alts.push_back(other + " " + nm(0));
            rule(nm(u), (end_ok ? std::string("| ") : std::string()) + vllm_join(alts, " | "));
        }
        return nm(0);
    }

    // a GBNF grammar (root rule required) imported under a private prefix
    std::string import(const std::string & g) {
        const std::string pre = fresh("g") + "-";
        struct def { std::string name; size_t body, start; };
        std::vector<def> defs;
        const size_t N = g.size();
        auto skip_quoted = [&](size_t i, char close) {
            ++i;
            while (i < N && g[i] != close) { i += g[i] == '\\' ? 2 : 1; }
            return std::min(N, i + 1);
        };
        for (size_t i = 0; i < N;) {
            const char c = g[i];
            if (c == '#') { while (i < N && g[i] != '\n') { ++i; } continue; }
            if (c == '"') { i = skip_quoted(i, '"'); continue; }
            if (c == '[') { i = skip_quoted(i, ']'); continue; }
            if (vllm_gbnf_word(c)) {
                const size_t s = i;
                while (i < N && vllm_gbnf_word(g[i])) { ++i; }
                size_t k = i;
                while (k < N && (g[k] == ' ' || g[k] == '\t')) { ++k; }
                if (g.compare(k, 3, "::=") == 0) {
                    defs.push_back({g.substr(s, i - s), k + 3, s});
                    i = k + 3;
                }
                continue;
            }
            ++i;
        }
        std::set<std::string> names;
        for (const auto & d : defs) { names.insert(d.name); }
        if (!names.count("root")) { throw std::runtime_error("structural_tag: grammar has no root rule"); }
        for (size_t d = 0; d < defs.size(); ++d) {
            const size_t end = d + 1 < defs.size() ? defs[d + 1].start : N;
            const std::string src = g.substr(defs[d].body, end - defs[d].body);
            std::string body;
            for (size_t i = 0; i < src.size();) {
                const char c = src[i];
                if (c == '#') { while (i < src.size() && src[i] != '\n') { ++i; } continue; }
                if (c == '"' || c == '[') {
                    const char close = c == '"' ? '"' : ']';
                    size_t j = i + 1;
                    while (j < src.size() && src[j] != close) { j += src[j] == '\\' ? 2 : 1; }
                    j = std::min(src.size(), j + 1);
                    body += src.substr(i, j - i);
                    i = j;
                    continue;
                }
                if (vllm_gbnf_word(c)) {
                    size_t j = i;
                    while (j < src.size() && vllm_gbnf_word(src[j])) { ++j; }
                    const std::string w = src.substr(i, j - i);
                    body += names.count(w) ? pre + w : w;
                    i = j;
                    continue;
                }
                body += (c == '\n' || c == '\r') ? ' ' : c;
                ++i;
            }
            rule(pre + defs[d].name, body);
        }
        return pre + "root";
    }

    static std::vector<std::string> ends_of(const json & f) {
        std::vector<std::string> out;
        const json & e = f.at("end");
        if (e.is_string()) { out.push_back(e.get<std::string>()); }
        else { for (const auto & x : e) { out.push_back(x.get<std::string>()); } }
        if (out.empty()) { throw std::runtime_error("structural_tag: tag end must be a string or a non-empty list"); }
        return out;
    }

    // any text up to (and including) the first of the end strings; "" = end of generation
    std::string any_until(const std::vector<std::string> & ends) {
        std::vector<std::u32string> w;
        std::vector<std::string> c;
        std::vector<bool> x;
        bool to_eos = false;
        for (const auto & e : ends) {
            if (e.empty()) { to_eos = true; continue; }
            w.push_back(vllm_u32(e));
            c.push_back("");
            x.push_back(false);
        }
        if (w.empty()) { return any_text(); }
        return scan(w, c, x, to_eos);
    }

    std::string until(const json & content, const std::vector<std::string> & ends) {
        const std::string ct = json_value(content, "type", std::string());
        if (ct == "any_text") { return any_until(ends); }
        if (ct == "sequence") {
            const json & el = content.at("elements");
            if (!el.empty() && json_value(el.back(), "type", std::string()) == "any_text") {
                std::vector<std::string> parts;
                for (size_t i = 0; i + 1 < el.size(); ++i) { parts.push_back(gen(el[i])); }
                parts.push_back(any_until(ends));
                return "(" + vllm_join(parts, " ") + ")";
            }
        }
        std::vector<std::string> alts;
        for (const auto & e : ends) { alts.push_back(lit(e)); }
        return "(" + gen(content) + " (" + vllm_join(alts, " | ") + "))";
    }

    std::string tag(const json & f) {
        return "(" + lit(f.at("begin").get<std::string>()) + " " + until(f.at("content"), ends_of(f)) + ")";
    }

    std::string triggered(const json & f, bool lazy) {
        std::vector<std::string> triggers;
        for (const auto & t : f.at("triggers")) { triggers.push_back(t.get<std::string>()); }
        if (triggers.empty()) { throw std::runtime_error("structural_tag: triggered_tags needs at least one trigger"); }
        const json & tags = f.at("tags");
        const bool at_least_one = json_value(f, "at_least_one", false);
        const bool stop_after_first = json_value(f, "stop_after_first", false);
        std::vector<std::string> excludes;
        for (const auto & x : json_value(f, "excludes", json::array())) { excludes.push_back(x.get<std::string>()); }
        for (const auto & tg : tags) {
            const std::string begin = tg.at("begin").get<std::string>();
            bool ok = false;
            for (const auto & t : triggers) { ok = ok || begin.rfind(t, 0) == 0; }
            if (!ok) { throw std::runtime_error("structural_tag: tag begin '" + begin + "' does not start with any trigger"); }
        }
        const std::string post = stop_after_first ? std::string() : fresh("free");
        std::vector<std::string> after;
        for (const auto & t : triggers) {
            std::vector<std::string> alts;
            for (const auto & tg : tags) {
                const std::string begin = tg.at("begin").get<std::string>();
                if (begin.rfind(t, 0) != 0) { continue; }
                const std::string rest = begin.substr(t.size());
                alts.push_back((rest.empty() ? std::string() : lit(rest) + " ") + until(tg.at("content"), ends_of(tg)));
            }
            if (alts.empty()) { throw std::runtime_error("structural_tag: trigger '" + t + "' starts no tag"); }
            after.push_back(rule(fresh("after"), "(" + vllm_join(alts, " | ") + ")" + (post.empty() ? std::string() : " " + post)));
        }
        std::vector<std::u32string> words;
        std::vector<std::string> cont;
        std::vector<bool> excl;
        for (size_t i = 0; i < triggers.size(); ++i) { words.push_back(vllm_u32(triggers[i])); cont.push_back(after[i]); excl.push_back(false); }
        for (const auto & x : excludes) { words.push_back(vllm_u32(x)); cont.push_back(""); excl.push_back(true); }
        if (!post.empty()) { rule(post, scan(words, cont, excl, true)); }
        if (lazy) {
            std::vector<std::string> alts;
            for (size_t i = 0; i < triggers.size(); ++i) { alts.push_back(lit(triggers[i]) + " " + after[i]); }
            return "(" + vllm_join(alts, " | ") + ")";
        }
        if (!at_least_one) { return post.empty() ? scan(words, cont, excl, true) : post; }
        return scan(words, cont, excl, false);
    }

    std::string gen(const json & f) {
        const std::string type = json_value(f, "type", std::string());
        if (type == "const_string") { return lit(f.at("value").get<std::string>()); }
        if (type == "json_schema") {
            docs.push_back(common_chat_schema_from_json(f.at("json_schema")));
            return b.add_schema(fresh("json"), *docs.back().root);
        }
        if (type == "regex") { return import(vllm_regex_grammar(f.at("pattern").get<std::string>())); }
        if (type == "grammar") {
            const std::string g = f.at("grammar").get<std::string>();
            if (g.find("::=") == std::string::npos) {
                throw std::runtime_error("structural_tag: only GBNF/EBNF ('::=') grammars run on llama.cpp; Lark grammars need llguidance");
            }
            return import(g);
        }
        if (type == "any_text") { return any_text(); }
        if (type == "sequence" || type == "or") {
            std::vector<std::string> parts;
            for (const auto & e : f.at("elements")) { parts.push_back(gen(e)); }
            if (parts.empty()) { return "\"\""; }
            return "(" + vllm_join(parts, type == "or" ? " | " : " ") + ")";
        }
        if (type == "tag") { return tag(f); }
        if (type == "triggered_tags") { return triggered(f, false); }
        if (type == "tags_with_separator") {
            std::vector<std::string> alts;
            for (const auto & tg : f.at("tags")) { alts.push_back(tag(tg)); }
            if (alts.empty()) { throw std::runtime_error("structural_tag: tags_with_separator needs at least one tag"); }
            const std::string one = "(" + vllm_join(alts, " | ") + ")";
            const bool at_least_one = json_value(f, "at_least_one", false);
            if (json_value(f, "stop_after_first", false)) { return at_least_one ? one : "(" + one + ")?"; }
            const std::string many = one + " (" + lit(f.at("separator").get<std::string>()) + " " + one + ")*";
            return at_least_one ? "(" + many + ")" : "(" + many + ")?";
        }
        if (type == "qwen_xml_parameter") {
            const json & schema = f.at("json_schema");
            std::set<std::string> required;
            for (const auto & r : json_value(schema, "required", json::array())) { required.insert(r.get<std::string>()); }
            const std::string ws = "[ \\t\\n]*";
            std::vector<std::string> parts;
            for (const auto & kv : json_value(schema, "properties", json::object()).items()) {
                const json & ps = kv.value();
                std::string val;
                if (json_value(ps, "type", std::string()) == "string") {
                    val = scan({vllm_u32("</parameter>")}, {""}, {false}, false);
                } else {
                    docs.push_back(common_chat_schema_from_json(ps));
                    val = b.add_schema(fresh("xmlp"), *docs.back().root) + " " + ws + " " + lit("</parameter>");
                }
                const std::string item = lit("<parameter=" + kv.key() + ">") + " " + ws + " " + val + " " + ws;
                parts.push_back(required.count(kv.key()) ? item : "(" + item + ")?");
            }
            return parts.empty() ? std::string("\"\"") : "(" + vllm_join(parts, " ") + ")";
        }
        throw std::runtime_error("structural_tag: unsupported format type '" + type + "'");
    }
};

static void vllm_structural(task_params & params, json st) {
    if (st.is_string()) { st = json::parse(st.get<std::string>()); }
    if (!st.is_object()) { throw std::runtime_error("structural_tag must be an object"); }
    json f;
    if (st.contains("format")) {
        f = st.at("format");
    } else if (st.contains("structures")) {
        json tags = json::array();
        for (const auto & s : st.at("structures")) {
            tags.push_back(json{{"type", "tag"}, {"begin", s.at("begin")},
                                {"content", json{{"type", "json_schema"}, {"json_schema", json_value(s, "schema", json::object())}}},
                                {"end", s.at("end")}});
        }
        f = json{{"type", "triggered_tags"}, {"triggers", json_value(st, "triggers", json::array())}, {"tags", tags}};
    } else {
        throw std::runtime_error("structural_tag: expected 'format' or 'structures' + 'triggers'");
    }
    const bool lazy = json_value(f, "type", std::string()) == "triggered_tags" && !json_value(f, "at_least_one", false)
                      && json_value(f, "excludes", json::array()).empty();
    const std::string g = build_grammar([&](const common_grammar_builder & b) {
        vllm_st_compiler c(b);
        const std::string root = lazy ? c.triggered(f, true) : c.gen(f);
        b.add_rule("root", root);
    });
    vllm_set_grammar(params, COMMON_GRAMMAR_TYPE_USER, g, "structured_outputs.structural_tag");
    params.vllm.st_grammar = g;
    params.vllm.st_lazy = lazy;
    params.vllm.st_triggers.clear();
    if (lazy) {
        for (const auto & t : f.at("triggers")) {
            params.vllm.st_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, t.get<std::string>()});
        }
    }
    params.sampling.grammar_lazy = lazy;
    params.sampling.grammar_triggers = params.vllm.st_triggers;
}

static void vllm_structured(task_params & params, json so) {
    static const char * kinds[] = {"json", "regex", "choice", "grammar", "json_object", "structural_tag"};
    std::string kind;
    for (const char * k : kinds) {
        if (so.contains(k) && !so.at(k).is_null() && !(so.at(k).is_boolean() && !so.at(k).get<bool>())) {
            if (!kind.empty()) { throw std::runtime_error("structured_outputs: exactly one of json, regex, choice, grammar, json_object, structural_tag"); }
            kind = k;
        }
    }
    if (kind.empty()) { return; }
    if (kind == "json") {
        json schema = so.at("json");
        if (schema.is_string()) { schema = json::parse(schema.get<std::string>()); }
        vllm_set_grammar(params, vllm_ws_custom(so) ? COMMON_GRAMMAR_TYPE_USER : COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, vllm_json_grammar(schema, so), "structured_outputs.json");
    } else if (kind == "json_object") {
        vllm_set_grammar(params, vllm_ws_custom(so) ? COMMON_GRAMMAR_TYPE_USER : COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, vllm_json_grammar(json{{"type", "object"}}, so), "structured_outputs.json_object");
    } else if (kind == "regex") {
        vllm_set_grammar(params, COMMON_GRAMMAR_TYPE_USER, vllm_regex_grammar(so.at("regex").get<std::string>()), "structured_outputs.regex");
    } else if (kind == "choice") {
        std::string g = "root ::= ";
        bool first = true;
        for (const auto & c : so.at("choice")) {
            g += (first ? "" : " | ") + vllm_gbnf_literal(c.get<std::string>());
            first = false;
        }
        if (first) { throw std::runtime_error("structured_outputs.choice must be a non-empty list"); }
        vllm_set_grammar(params, COMMON_GRAMMAR_TYPE_USER, g, "structured_outputs.choice");
    } else if (kind == "grammar") {
        const std::string g = so.at("grammar").get<std::string>();
        if (g.find("::=") == std::string::npos) {
            throw std::runtime_error("structured_outputs.grammar: only GBNF/EBNF ('::=') grammars run on llama.cpp; Lark grammars need llguidance");
        }
        vllm_set_grammar(params, COMMON_GRAMMAR_TYPE_USER, g, "structured_outputs.grammar");
    } else {
        vllm_structural(params, so.at("structural_tag")); // local patch 0055
    }
}

std::vector<std::unique_ptr<field>> make_llama_cmpl_schema(const common_params & params_base, task_params & params) {
    std::vector<std::unique_ptr<field>> fields;
    auto add = [&](field * f) {
        fields.emplace_back(f);
    };

    add((new field_bool("verbose", params.verbose))
        ->set_desc("Include __verbose field in the response with additional debug information"));

    add((new field_bool("timings_per_token", params.timings_per_token))
        ->set_desc("Include prompt processing and text generation speed information in each response"));

    add((new field_bool("stream", params.stream))
        ->set_desc("Allows receiving each predicted token in real-time instead of waiting for the completion to finish"));

    add((new field_nested("stream_options"))
        ->add_subfield((new field_bool("include_usage", params.include_usage))
            ->set_desc("Whether to include usage information in the stream"))
        ->set_desc("Additional options for streaming responses"));

    add((new field_bool("cache_prompt", params.cache_prompt))
        ->set_desc("Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests"));

    add((new field_bool("return_tokens", params.return_tokens))
        ->set_desc("Return the raw generated token ids in the `tokens` field"));

    add((new field_bool("return_progress", params.return_progress))
        ->set_desc("Include prompt processing progress events in stream mode"));

    add((new field_num("sse_ping_interval", params.sse_ping_interval))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Interval in seconds between SSE comment pings emitted while the stream stays silent, -1 disables pings"));

    add((new field_num("n_predict", params.n_predict))
        ->set_hard_limits(-1, INT32_MAX)
        ->add_alias("max_completion_tokens")
        ->add_alias("max_tokens")
        ->set_desc("Set the maximum number of tokens to predict. When 0, no tokens will be generated but the prompt is evaluated into the cache"));

    add((new field_num("n_indent", params.n_indent))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Specify the minimum line indentation for the generated text in number of whitespace characters. Useful for code completion tasks"));

    add((new field_num("n_keep", params.n_keep))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Specify the number of tokens from the initial prompt to retain when context size is exceeded. Use -1 to retain all tokens from the prompt"));

    add((new field_num("n_discard", params.n_discard))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Number of tokens after n_keep that may be discarded when shifting context (0 = half context)"));

    add((new field_num("n_cmpl", params.n_cmpl))
        ->set_hard_limits(1, params_base.n_parallel)
        ->add_alias("n") // alias "n" as fallback (OpenAI completions API)
        ->set_desc("Number of completions to generate. If the input has multiple prompts, total outputs will be N prompts times n_cmpl"));

    add((new field_num("n_cache_reuse", params.n_cache_reuse))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Min chunk size to attempt reusing from the cache via KV shifting. See --cache-reuse arg"));

    // TODO: implement t_max_prompt_ms
    // add((new field_num("t_max_prompt_ms", params.t_max_prompt_ms))

    add((new field_num("t_max_predict_ms", params.t_max_predict_ms))
        ->set_hard_limits(-1, std::numeric_limits<int64_t>::max())
        ->set_desc("Set a time limit in milliseconds for the prediction phase. The timeout triggers if generation exceeds this time (measured since the first token) and a newline has been generated. Useful for FIM applications"));

    add((new field_json("response_fields"))
        ->set_desc("A list of response fields to return. Missing fields are omitted without error. Fields with a slash are unnested (e.g. generation_settings/n_predict moves n_predict to the root)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.response_fields = json_value(data, "response_fields", std::vector<std::string>());
        }));


    //
    // Sampling params
    //

    add((new field_num("top_k", params.sampling.top_k))
        ->set_limits(0, INT32_MAX)
        ->set_desc("Limit the next token selection to the K most probable tokens (0 = disabled)"));

    add((new field_num("top_p", params.sampling.top_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Limit the next token selection to a subset of tokens with cumulative probability above threshold P (1.0 = disabled)"));

    add((new field_num("min_p", params.sampling.min_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("The minimum probability for a token to be considered, relative to the probability of the most likely token (0 = disabled)"));

    add((new field_num("top_n_sigma", params.sampling.top_n_sigma))
        ->set_desc("Keep tokens within n standard deviations of the top token logit (< 0 = disabled)"));

    add((new field_num("xtc_probability", params.sampling.xtc_probability))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Set the chance for token removal via XTC sampler (0 = disabled)"));

    add((new field_num("xtc_threshold", params.sampling.xtc_threshold))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Set a minimum probability threshold for tokens to be removed via XTC sampler (> 0.5 disables XTC)"));

    add((new field_num("typical_p", params.sampling.typ_p))
        // ->set_limits(0.0f, 1.0f) // what's the valid range?
        ->set_desc("Enable locally typical sampling with parameter p (1.0 = disabled)"));

    add((new field_num("temperature", params.sampling.temp))
        ->set_limits(0.0f, std::numeric_limits<float>::infinity())
        ->set_desc("Adjust the randomness of the generated text (0 = greedy)"));

    add((new field_num("dynatemp_range", params.sampling.dynatemp_range))
        ->set_desc("Dynamic temperature range. The final temperature will be in [temperature - range, temperature + range] (0 = disabled)"));

    add((new field_num("dynatemp_exponent", params.sampling.dynatemp_exponent))
        ->set_desc("Dynamic temperature exponent, controls how entropy maps to temperature"));

    add((new field_num("repeat_last_n", params.sampling.penalty_last_n))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Last n tokens to consider for penalizing repetition (0 = disabled)"));

    add((new field_num("repeat_penalty", params.sampling.penalty_repeat))
        ->set_desc("Control the repetition of token sequences in the generated text (1.0 = disabled)"));

    add((new field_num("frequency_penalty", params.sampling.penalty_freq))
        ->set_desc("Repeat alpha frequency penalty (0 = disabled)"));

    add((new field_num("presence_penalty", params.sampling.penalty_present))
        ->set_desc("Repeat alpha presence penalty (0 = disabled)"));

    add((new field_num("dry_multiplier", params.sampling.dry_multiplier))
        ->set_desc("Set the DRY (Don't Repeat Yourself) repetition penalty multiplier (0 = disabled)"));

    add((new field_num("dry_base", params.sampling.dry_base))
        ->set_desc("Set the DRY repetition penalty base value (must be >= 1.0, any values < 1.0 will be replaced with the default value)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            float v = data.at("dry_base").get<float>();
            ctx.params.sampling.dry_base = (v < 1.0f) ? params_base.sampling.dry_base : v;
        }));

    add((new field_num("dry_allowed_length", params.sampling.dry_allowed_length))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Tokens that extend repetition beyond this length receive exponentially increasing penalty: multiplier * base ^ (sequence_length - allowed_length)"));

    add((new field_num("dry_penalty_last_n", params.sampling.dry_penalty_last_n))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("How many tokens to scan for repetitions (0 = disabled)"));

    add((new field_num("mirostat", params.sampling.mirostat))
        ->set_limits(0, 2)
        ->set_desc("Enable Mirostat sampling, controlling perplexity during text generation (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"));

    add((new field_num("mirostat_tau", params.sampling.mirostat_tau))
        ->set_desc("Set the Mirostat target entropy, parameter tau"));

    add((new field_num("mirostat_eta", params.sampling.mirostat_eta))
        ->set_desc("Set the Mirostat learning rate, parameter eta"));

    add((new field_num("adaptive_target", params.sampling.adaptive_target))
        ->set_limits(-std::numeric_limits<float>::max(), 1.0f)
        ->set_desc("Adaptive sampling target entropy (valid range 0.0 to 1.0; negative = disabled)"));

    add((new field_num("adaptive_decay", params.sampling.adaptive_decay))
        ->set_hard_limits(0.0f, 0.99f)
        ->set_desc("EMA decay for adaptive sampling; history approximates 1/(1-decay) tokens"));

    // seed is uint32_t; field_num uses int32_t so use a handler
    add((new field_num("seed", params.sampling.seed))
        ->set_desc("Set the random number generator (RNG) seed (-1 = random)"));

    add((new field_num("n_probs", params.sampling.n_probs))
        ->add_alias("logprobs") // use "logprobs" if "n_probs" wasn't provided
        ->set_desc("If greater than 0, output the probabilities of top N tokens for each generated token"));

    add((new field_num("min_keep", params.sampling.min_keep))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("If greater than 0, force samplers to return at least N possible tokens"));

    add((new field_bool("backend_sampling", params.sampling.backend_sampling))
        ->set_desc("Use backend sampling instead of llama.cpp sampling"));

    add((new field_bool("post_sampling_probs", params.post_sampling_probs))
        ->set_desc("Return probabilities of top n_probs tokens after applying the sampling chain"));

    //
    // Speculative decoding params
    //

    // TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
    // TODO: for now, be able to adjust only the draft-model based speculative parameters
    add((new field_num("speculative.n_max", params.speculative.draft.n_max))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Maximum number of tokens to draft during speculative decoding"));

    add((new field_num("speculative.n_min", params.speculative.draft.n_min))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Minimum number of draft tokens to use for speculative decoding");

    add((new field_num("speculative.p_min", params.speculative.draft.p_min))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("Minimum speculative decoding probability for draft tokens (0 = greedy)"));


    add((new field_str("speculative.type"))
        ->set_desc("Speculative decoding method (for debugging and research purposes)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.speculative.types = { common_speculative_type_from_name(data.at("speculative.type").get<std::string>()) };
        }));

    add((new field_num("speculative.ngram_size_n", params.speculative.ngram_simple.size_n))
        ->set_desc("Ngram size for lookup in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_size_m", params.speculative.ngram_simple.size_m))
        ->set_desc("Mgram size for speculative tokens in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_min_hits", params.speculative.ngram_simple.min_hits))
        ->set_desc("Minimum hits at ngram lookup for mgram to be proposed"));
#endif

    add((new field_json("lora"))
        ->set_desc("A list of LoRA adapters to apply to this request. Each entry must have `id` and `scale` fields. Adapters not listed default to scale 0.0")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & lora = data.at("lora");
            if (!lora.is_array()) {
                throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
            }
            ctx.params.lora = parse_lora_request(lora);
        }));

    // sequence breakers for DRY
    // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
    // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39
    add((new field_json("dry_sequence_breakers"))
        ->set_desc("Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (ctx.params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }

            // each distinct breaker costs a full vocabulary scan at sampler init, so dedupe before counting
            std::unordered_set<std::string> seen;
            std::vector<std::string> unique_breakers;

            for (const auto & breaker : ctx.params.sampling.dry_sequence_breakers) {
                if (seen.insert(breaker).second) {
                    unique_breakers.push_back(breaker);
                }
            }

            if (unique_breakers.size() > LLAMA_DRY_MAX_SEQ_BREAKERS) {
                throw std::runtime_error(string_format(
                    "Error: dry_sequence_breakers must contain at most %d distinct entries", LLAMA_DRY_MAX_SEQ_BREAKERS));
            }

            ctx.params.sampling.dry_sequence_breakers = std::move(unique_breakers);
        }));

    // handle both "json_schema" and "grammar"
    add((new field_json("json_schema"))
        ->add_alias("grammar")
        ->set_desc("Set a JSON schema (json_schema) or GBNF grammar string (grammar) for constrained generation. json_schema takes precedence if both are provided")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            auto & params = ctx.params;
            if (data.contains("json_schema") && !data.contains("grammar")) {
                try {
                    auto schema                  = json_value(data, "json_schema", json::object());
                    if (schema.is_object() && schema.empty()) {
                        // an empty schema means any object
                        schema["type"] = "object";
                    }
                    SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
                    std::string grammar_str      = json_schema_to_grammar(schema);
                    SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
                    params.sampling.grammar      = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str)};
                } catch (const std::exception & e) {
                    throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
                }
            } else {
                std::string grammar_str = json_value(data, "grammar", std::string());
                if (!grammar_str.empty()) {
                    // grammar_type key is set by the server when converting chat template grammars
                    std::string grammar_type = json_value(data, "grammar_type", std::string());
                    if (grammar_type == "tool_calls") {
                        params.sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str)};
                    } else {
                        // explicit grammar from the user (API field "grammar")
                        params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str)};
                    }
                    SRV_DBG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(params.sampling.grammar).c_str());
                }
            }
        }));

    add((new field_bool("grammar_lazy", params.sampling.grammar_lazy))
        ->set_desc("Whether to apply grammar constraints lazily, only when triggered (instead of at every step)"));

    //
    // Chat parser params
    //

    add((new field_str("reasoning_format"))
        ->set_desc("Reasoning format for chain-of-thought models")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
        }));

    add((new field_json("continue_final_message"))
        ->set_desc("Whether to continue the final message of the chat template")
        ->set_handler([&](field_eval_context &, const json & data) {
            common_chat_continuation_parse(data.at("continue_final_message"));
        }));

    add((new field_json("echo"))
        ->set_desc("Whether to include the continued assistant message in the output")
        ->set_handler([&](field_eval_context &, const json & data) {
            data.at("echo").get<bool>();
        }));

    //
    // Token-level fields (require vocab)
    //

    add((new field_json("preserved_tokens"))
        ->set_desc("List of token strings that must not be split during tokenization")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            common_sampling_add_preserved_tokens(ctx.params.sampling, ctx.vocab, data.at("preserved_tokens").get<std::vector<std::string>>());
        }));

    add((new field_json("grammar_triggers"))
        ->set_desc("List of strings or patterns that trigger grammar-constrained generation")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            std::vector<common_grammar_trigger> triggers;
            for (const auto & t : data.at("grammar_triggers")) {
                triggers.push_back(server_grammar_trigger(t).value);
            }
            common_sampling_add_grammar_triggers(ctx.params.sampling, ctx.vocab, std::move(triggers));
        }));

    add((new field_bool("reasoning_control", params.sampling.reasoning_control))
        ->set_desc("Create the budget sampler on demand so reasoning can be ended at runtime"));

    add((new field_num("reasoning_budget_tokens", params.sampling.reasoning_budget_tokens))
        ->set_hard_limits(-1, INT32_MAX)
        ->set_desc("Number of tokens in the reasoning budget (-1 = disabled)"));

    add((new field_str("reasoning_budget_start_tag"))
        ->set_desc("Token string marking the start of the reasoning budget section")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.reasoning_budget_start = common_tokenize(ctx.vocab, data.at("reasoning_budget_start_tag").get<std::string>(), false, true);
        }));

    add((new field_json("reasoning_budget_end_tags"))
        ->add_alias("reasoning_budget_end_tag")
        ->set_desc("Token strings marking the end of the reasoning budget section; the first is forced when the budget expires")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.reasoning_budget_end.clear();
            if (data.contains("reasoning_budget_end_tags")) {
                for (const auto & t : data.at("reasoning_budget_end_tags")) {
                    std::string tag = t.get<std::string>();
                    if (!tag.empty()) {
                        ctx.params.sampling.reasoning_budget_end.push_back(common_tokenize(ctx.vocab, tag, false, true));
                    }
                }
            } else if (data.contains("reasoning_budget_end_tag")) {
                std::string tag = data.at("reasoning_budget_end_tag").get<std::string>();
                if (!tag.empty()) {
                    ctx.params.sampling.reasoning_budget_end.push_back(common_tokenize(ctx.vocab, tag, false, true));
                }
            }
        }));

    add((new field_str("reasoning_budget_message"))
        ->set_desc("Message to prepend to the reasoning budget end tag when forcing it")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            if (!ctx.params.sampling.reasoning_budget_end.empty()) {
                llama_tokens end_tag = ctx.params.sampling.reasoning_budget_end.front();
                std::string message = json_value(data, "reasoning_budget_message", std::string());
                if (!message.empty()) {
                    llama_tokens message_tokens = common_tokenize(ctx.vocab, message, false, true);
                    end_tag.insert(end_tag.begin(), message_tokens.begin(), message_tokens.end());
                }
                ctx.params.sampling.reasoning_budget_forced = std::move(end_tag);
            }
        }));

    add((new field_json("logit_bias"))
        ->set_desc("Modify the likelihood of specific tokens. Accepts an array of [token, bias] pairs or an object mapping token to bias. Use false as bias to ban a token")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.vocab != nullptr);
            ctx.params.sampling.logit_bias.clear();
            const auto & logit_bias = data.at("logit_bias");
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            auto parse_bias = [](const json & v, float & bias) -> bool {
                if (v.is_number())                        { bias = v.get<float>(); return true; }
                if (v.is_boolean() && !v.get<bool>())     { bias = -INFINITY;      return true; }
                return false;
            };
            if (logit_bias.is_array()) {
                for (const auto & el : logit_bias) {
                    if (!el.is_array() || el.size() != 2) continue;
                    float bias;
                    if (!parse_bias(el[1], bias)) continue;
                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) ctx.params.sampling.logit_bias.push_back({tok, bias});
                    } else if (el[0].is_string()) {
                        for (auto tok : common_tokenize(ctx.vocab, el[0].get<std::string>(), false))
                            ctx.params.sampling.logit_bias.push_back({tok, bias});
                    }
                }
            } else if (logit_bias.is_object()) {
                for (const auto & el : logit_bias.items()) {
                    float bias;
                    if (!parse_bias(el.value(), bias)) continue;
                    char * end;
                    llama_token tok = strtol(el.key().c_str(), &end, 10);
                    if (*end == 0) {
                        if (tok >= 0 && tok < n_vocab) ctx.params.sampling.logit_bias.push_back({tok, bias});
                    } else {
                        for (auto t : common_tokenize(ctx.vocab, el.key(), false))
                            ctx.params.sampling.logit_bias.push_back({t, bias});
                    }
                }
            }
        }));

    add((new field_bool("ignore_eos", params.sampling.ignore_eos))
        ->set_desc("Ignore the end-of-sequence token and continue generating")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            GGML_ASSERT(ctx.logit_bias_eog != nullptr);
            ctx.params.sampling.ignore_eos = data.at("ignore_eos").get<bool>();
            if (ctx.params.sampling.ignore_eos && ctx.logit_bias_eog) {
                ctx.params.sampling.logit_bias.insert(
                    ctx.params.sampling.logit_bias.end(),
                    ctx.logit_bias_eog->begin(), ctx.logit_bias_eog->end());
            }
        }));

    add((new field_json("stop"))
        ->set_desc("Specify stopping strings. Generation stops when one is produced, and the string is not included in the output")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.antiprompt.clear();
            const auto & stop = data.at("stop");
            if (stop.is_array()) {
                for (const auto & word : stop) {
                    if (!word.empty()) ctx.params.antiprompt.push_back(word);
                }
            } else if (stop.is_string()) {
                ctx.params.antiprompt.push_back(stop.get<std::string>());
            }
            // fall back to CLI defaults if the request provided no effective stop strings
            if (ctx.params.antiprompt.empty()) {
                ctx.params.antiprompt = params_base.antiprompt;
            }
        }));

    add((new field_json("samplers"))
        ->set_desc("The order in which samplers are applied. An array of sampler type names, or a single string of sampler chars")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & samplers = data.at("samplers");
            if (samplers.is_array()) {
                ctx.params.sampling.samplers = common_sampler_types_from_names(samplers.get<std::vector<std::string>>());
            } else if (samplers.is_string()) {
                ctx.params.sampling.samplers = common_sampler_types_from_chars(samplers.get<std::string>());
            }
        }));

    // ---- local patch 0053 (engine parity): vLLM request controls, vLLM semantics ----
    add((new field_json("min_tokens"))
        ->set_desc("vLLM: minimum output tokens before EOS, stop_token_ids or stop strings can end the generation")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const int n = data.at("min_tokens").get<int>();
            if (n < 0) { throw std::runtime_error("min_tokens must be >= 0"); }
            ctx.params.vllm.min_tokens = n;
        }));
    add((new field_json("stop_token_ids"))
        ->set_desc("vLLM: token ids that end the generation")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("stop_token_ids");
            if (v.is_null()) { return; }
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            for (const auto & t : v) {
                const int id = t.get<int>();
                if (id < 0 || id >= n_vocab) { throw std::runtime_error("stop_token_ids: token id out of range"); }
                ctx.params.vllm.stop_token_ids.push_back(id);
            }
        }));
    add((new field_bool("include_stop_str_in_output", params.vllm.include_stop_str_in_output))
        ->set_desc("vLLM: keep the matched stop string in the output"));
    add((new field_bool("skip_special_tokens", params.vllm.skip_special_tokens))
        ->set_desc("vLLM: false renders special tokens in the output"));
    add((new field_bool("spaces_between_special_tokens", params.vllm.spaces_between_special_tokens))
        ->set_desc("vLLM: join rendered special tokens and the text around them with spaces"));
    add((new field_bool("include_reasoning", params.vllm.include_reasoning))
        ->set_desc("vLLM: false drops the reasoning from chat output"));
    add((new field_bool("return_token_ids", params.vllm.return_token_ids))
        ->set_desc("vLLM: return the generated token ids in choices[].token_ids"));
    add((new field_json("bad_words"))
        ->set_desc("vLLM: words that may not be generated (the word and its leading-space form; last token banned after a prefix match)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("bad_words");
            if (v.is_null()) { return; }
            for (const auto & w : v) {
                const std::string s = w.get<std::string>();
                for (const std::string & variant : {s, std::string(" ") + s}) {
                    llama_tokens toks = common_tokenize(ctx.vocab, variant, false, false);
                    if (!toks.empty()) { ctx.params.vllm.bad_words.push_back(std::move(toks)); }
                }
            }
        }));
    add((new field_json("allowed_token_ids"))
        ->set_desc("vLLM: only these token ids may be generated")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("allowed_token_ids");
            if (v.is_null()) { return; }
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            std::vector<char> ok(n_vocab, 0);
            for (const auto & t : v) {
                const int id = t.get<int>();
                if (id < 0 || id >= n_vocab) { throw std::runtime_error("allowed_token_ids: token id out of range"); }
                ok[id] = 1;
            }
            for (int id = 0; id < n_vocab; ++id) {
                if (!ok[id]) { ctx.params.sampling.logit_bias.push_back({id, -INFINITY}); }
            }
        }));
    add((new field_json("repetition_penalty"))
        ->set_desc("vLLM: repetition penalty over the whole sequence (prompt + output)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            if (data.at("repetition_penalty").is_null()) { return; }
            const float v = data.at("repetition_penalty").get<float>();
            if (data.contains("repeat_penalty") && data.at("repeat_penalty").get<float>() != v) {
                throw std::runtime_error("Conflicting controls: repetition_penalty and repeat_penalty");
            }
            ctx.params.sampling.penalty_repeat = v;
            if (v != 1.0f) { ctx.params.sampling.penalty_last_n = std::max(params_base.n_ctx, 1); }
        }));
    add((new field_json("cache_salt"))
        ->set_desc("vLLM: isolates the prefix cache; here the request reuses no cached prefix")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            if (!data.at("cache_salt").is_null()) { ctx.params.cache_prompt = false; }
        }));
    add((new field_json("truncate_prompt_tokens"))
        ->set_desc("vLLM: keep only the last K prompt tokens (-1 = context size)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            if (data.at("truncate_prompt_tokens").is_null()) { return; }
            const int k = data.at("truncate_prompt_tokens").get<int>();
            if (k < -1 || k == 0) { throw std::runtime_error("truncate_prompt_tokens must be -1 or >= 1"); }
            ctx.params.vllm.truncate_prompt_tokens = k;
            ctx.params.vllm.truncate_right = json_value(data, "truncation_side", std::string("left")) == "right";
        }));
    add((new field_json("structured_outputs"))
        ->set_desc("vLLM: json / regex / choice / grammar / json_object / structural_tag constraints")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            if (data.at("structured_outputs").is_null()) { return; }
            vllm_structured(ctx.params, data.at("structured_outputs"));
        }));
    // local patch 0054
    add((new field_json("repetition_detection"))
        ->set_desc("vLLM: end generation when an N-gram repeats min_count times at the end of the output")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & r = data.at("repetition_detection");
            if (r.is_null()) { return; }
            const int mx = json_value(r, "max_pattern_size", 0), mn = json_value(r, "min_pattern_size", 0), c = json_value(r, "min_count", 0);
            if (mx < 0 || mn < 0 || mn > mx) { throw std::runtime_error("max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size"); }
            if (mx > 0 && c < 2) { throw std::runtime_error("min_count must be >= 2 to detect repetitive patterns"); }
            ctx.params.vllm.rep_max = mx; ctx.params.vllm.rep_min = mn; ctx.params.vllm.rep_count = c;
        }));
    add((new field_num("stream_interval", params.vllm.stream_interval))
        ->set_hard_limits(1, INT32_MAX)
        ->set_desc("vLLM: stream one chunk per N generated tokens"));
    add((new field_bool("return_prompt_text", params.vllm.return_prompt_text))
        ->set_desc("vLLM: include the templated prompt as prompt_text"));
    add((new field_bool("return_tokens_as_token_ids", params.vllm.return_tokens_as_token_ids))
        ->set_desc("vLLM: render logprob tokens as token_id:N"));
    add((new field_json("logprob_token_ids"))
        ->set_desc("vLLM: return logprobs for exactly these token ids at each generated position (requires logprobs)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("logprob_token_ids");
            if (v.is_null() || v.empty()) { return; }
            if (!json_value(data, "logprobs", false) && json_value(data, "n_probs", 0) <= 0) {
                throw std::runtime_error("when using `logprob_token_ids`, `logprobs` must be set to true.");
            }
            const int n_vocab = llama_vocab_n_tokens(ctx.vocab);
            for (const auto & t : v) {
                const int id = t.get<int>();
                if (id < 0 || id >= n_vocab) { throw std::runtime_error("logprob_token_ids: token id out of range"); }
                ctx.params.vllm.logprob_token_ids.push_back(id);
            }
            if (ctx.params.sampling.n_probs <= 0) { ctx.params.sampling.n_probs = (int) ctx.params.vllm.logprob_token_ids.size(); }
        }));
    // local patch 0055
    add((new field_json("prompt_logprobs"))
        ->set_desc("vLLM: logprobs of every prompt token plus the top-k (-1 = whole vocabulary)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            const auto & v = data.at("prompt_logprobs");
            if (v.is_null()) { return; }
            const int k = v.get<int>();
            if (k < -1) { throw std::runtime_error("`prompt_logprobs` must be a positive value or -1."); }
            if (json_value(data, "stream", false)) { throw std::runtime_error("`prompt_logprobs` are not available when `stream=True`."); }
            const int eff = k == -1 ? llama_vocab_n_tokens(ctx.vocab) : k; // vLLM max_logprobs (20 on our engine)
            if (eff > 20) { throw std::runtime_error("Requested prompt logprobs of " + std::to_string(eff) + ", which is greater than max allowed: 20"); }
            ctx.params.vllm.prompt_logprobs = k;
        }));
    for (const char * g : {"guided_json", "guided_regex", "guided_choice", "guided_grammar"}) {
        add((new field_json(g))
            ->set_desc("vLLM (legacy name of structured_outputs)")
            ->set_handler([&, g](field_eval_context & ctx, const json & data) {
                if (data.at(g).is_null()) { return; }
                json so = json::object();
                so[std::string(g).substr(7)] = data.at(g);
                for (const char * opt : {"guided_whitespace_pattern"}) {
                    if (data.contains(opt)) { so["whitespace_pattern"] = data.at(opt); }
                }
                vllm_structured(ctx.params, so);
            }));
    }

    return fields;
}

task_params eval_llama_cmpl_schema(
                const llama_vocab * vocab,
                const common_params & params_base,
                const std::vector<llama_logit_bias> & logit_bias_eog,
                const json & data) {
    task_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
    params.sampling      = params_base.sampling;
    params.speculative   = params_base.speculative;
    params.n_keep        = params_base.n_keep;
    params.n_predict     = params_base.n_predict;
    params.n_cache_reuse = params_base.n_cache_reuse;
    params.cache_prompt  = params_base.cache_prompt;
    params.antiprompt    = params_base.antiprompt;
    params.sse_ping_interval = params_base.sse_ping_interval;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose       = params_base.verbosity > 9;

    params.reasoning_format = params_base.reasoning_format;

    // create context and schema
    field_eval_context ctx(params);
    ctx.vocab          = vocab;
    ctx.logit_bias_eog = &logit_bias_eog;

    auto schema = make_llama_cmpl_schema(params_base, params);

    // eval all fields in the schema
    for (const auto & f : schema) {
        f->eval(ctx, data);
    }

    // local patch 0053: host-side logit edits (min_tokens, bad_words) need host logits
    if (params.vllm.logits_active()) {
        params.sampling.backend_sampling = false;
        if (params.vllm.min_tokens > 0) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (int id = 0; id < n_vocab; ++id) {
                if (llama_vocab_is_eog(vocab, id)) { params.vllm.eog_ids.push_back(id); }
            }
        }
    }

    // debugging
    {
        auto budget = params.sampling.reasoning_budget_tokens;
        SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu seqs, forced=%zu toks\n",
                budget, params.sampling.generation_prompt.c_str(),
                params.sampling.reasoning_budget_start.size(),
                params.sampling.reasoning_budget_end.size(),
                params.sampling.reasoning_budget_forced.size());
    }

    return params;
}

//
// TTS schema
//

std::vector<std::unique_ptr<field>> make_tts_schema(server_task & task) {
    std::vector<std::unique_ptr<field>> fields;
    auto add = [&](field * f) {
        fields.emplace_back(f);
    };

    add((new field_num("top_k", task.tts_inp.data.top_k))
        ->set_limits(0, INT32_MAX)
        ->set_desc("Top-k for the acoustic code predictor, 0 to use the model default"));

    add((new field_num("top_p", task.tts_inp.data.top_p))
        ->set_limits(0.0f, 1.0f)
        ->set_desc("Top-p for the acoustic code predictor, 0.0 to use the model default"));

    add((new field_num("seed", task.tts_inp.data.seed))
        ->set_desc("RNG seed for the backbone sampler and the codec/vocoder (-1 = random)"));

    add((new field_num("n_predict", task.params.n_predict))
        ->set_hard_limits(-1, INT32_MAX)
        ->add_alias("max_tokens")
        ->set_desc("Max number of audio frames to generate"));

    add((new field_num("repeat_penalty", task.params.sampling.penalty_repeat))
        ->set_desc("Repetition penalty applied to the backbone sampler over the whole generation"));

    add((new field_bool("stream", task.params.stream))
        ->set_desc("Stream the audio as it becomes available instead of waiting for the full generation"));

    return fields;
}

void eval_tts_schema(const common_params & params_base, server_task & task, const json & data) {
    // baseline defaults, individual requests can override them
    // TODO @ngxson : change the default values based on model
    task.params.sampling = params_base.sampling;
    task.params.sampling.penalty_repeat = 1.05f;
    task.params.n_predict = -1;
    task.tts_inp.data.top_k = 0;
    task.tts_inp.data.top_p = 0.0f;
    task.tts_inp.data.seed  = params_base.sampling.seed;

    field_eval_context ctx(task.params);

    for (const auto & f : make_tts_schema(task)) {
        f->eval(ctx, data);
    }
}

//
// eval() implementations
//

static void handle_with_catch(const char * name, std::function<void()> func) {
    try {
        func();
    } catch (const std::exception & e) {
        throw std::invalid_argument(string_format("Field '%s': %s", name, e.what()));
    }
}

// treat a null value as absent so clients can send null to request the server default
static bool has_value(const json & data, const char * n) {
    return data.contains(n) && !data.at(n).is_null();
}

template <typename T>
void field_num<T>::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                if (custom_handler) {
                custom_handler(ctx, data);
                } else if (!is_hard_limit) {
                    val = std::max(min, std::min(max, data.at(n).template get<T>()));
                } else {
                    T tmp = data.at(n).template get<T>();
                    if (tmp < min || tmp > max) {
                        throw std::invalid_argument(std::string("Value must be between ") + std::to_string(min) + " <= value <= " + std::to_string(max) + ", but got " + std::to_string(tmp));
                    }
                    val = tmp;
                }
            });
            return;
        }
    }
}

void field_str::eval(field_eval_context & ctx, const json & data) {
    GGML_ASSERT(custom_handler);
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                custom_handler(ctx, data);
            });
            return;
        }
    }
}

void field_bool::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                if (custom_handler) {
                    custom_handler(ctx, data);
                } else {
                    val = data.at(n).get<bool>();
                }
            });
            return;
        }
    }
}

void field_json::eval(field_eval_context & ctx, const json & data) {
    GGML_ASSERT(custom_handler);
    for (const auto & n : name) {
        if (has_value(data, n)) {
            handle_with_catch(n, [&]() {
                custom_handler(ctx, data);
            });
            return;
        }
    }
}

void field_nested::eval(field_eval_context & ctx, const json & data) {
    for (const auto & n : name) {
        if (data.contains(n) && data.at(n).is_object()) {
            for (auto & f : subfields) {
                f->eval(ctx, data.at(n));
            }
            return;
        }
    }
}

} // namespace server_schema
