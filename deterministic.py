import ast
import json
import operator as op
import re
from datetime import datetime, timedelta, date

SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg,
}

def _safe_eval(expr):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("unsupported")
    try:
        val = walk(tree)
        if val == int(val):
            return str(int(val))
        return str(val)
    except Exception:
        return None

def solve_math(prompt):
    match = re.search(r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+)", prompt)
    if not match:
        return None
    result = _safe_eval(match.group(1))
    if result is None:
        return None
    return result

URL_PATTERN = re.compile(r'https?://[^\s)\]}>"\']+')
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

def solve_extract(prompt):
    emails = EMAIL_PATTERN.findall(prompt)
    urls = URL_PATTERN.findall(prompt)
    if emails and not urls:
        return emails[0].rstrip(".,;")
    if urls and not emails:
        return urls[0].rstrip(".,;")
    if emails and urls:
        return json.dumps({"emails": emails[:3], "urls": urls[:3]})
    return None

def solve_json(prompt):
    brace_match = re.search(r"(\{.*\}|\[.*\])", prompt, flags=re.DOTALL)
    if not brace_match:
        return None
    candidate = brace_match.group(1)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, (dict, list)):
        return json.dumps(obj)
    return None

SENTIMENT_LEXICON = {
    "positive": {"love", "excellent", "amazing", "great", "wonderful", "fantastic", "outstanding", "delightful", "beautiful", "perfect", "happy", "good", "impressed", "praise", "brilliant"},
    "negative": {"terrible", "awful", "horrible", "hate", "disaster", "worst", "poor", "bad", "broken", "crash", "fail", "failed", "fails", "painfully", "slow", "disappointed", "frustrating", "useless"},
}

SENTIMENT_NEGATORS = {"not", "no", "never", "neither", "nor", "nothing", "hardly", "barely", "doesn't", "don't", "didn't", "wasn't", "isn't"}

def solve_sentiment(prompt):
    lower = prompt.lower()
    words = set(re.findall(r"[a-zA-Z']+", lower))
    pos_score = sum(1 for w in words & SENTIMENT_LEXICON["positive"])
    neg_score = sum(1 for w in words & SENTIMENT_LEXICON["negative"])
    sentence_words = re.findall(r"[a-zA-Z']+", lower)
    for i, w in enumerate(sentence_words):
        if w in SENTIMENT_NEGATORS and i + 1 < len(sentence_words):
            nxt = sentence_words[i + 1]
            if nxt in SENTIMENT_LEXICON["positive"]:
                pos_score -= 1
                neg_score += 1
            elif nxt in SENTIMENT_LEXICON["negative"]:
                neg_score -= 1
                pos_score += 1
    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    if "classify the sentiment" in lower or "sentiment" in lower:
        return "neutral"
    return None

_STRING_PATTERNS = [
    (re.compile(r"(?:uppercase|upper|capitalize)\s+(?:of\s+)?['\"]?(\w+)['\"]?", re.I), lambda m: m.group(1).upper()),
    (re.compile(r"(?:lowercase|lower)\s+(?:of\s+)?['\"]?(\w+)['\"]?", re.I), lambda m: m.group(1).lower()),
    (re.compile(r"(?:reverse)\s+(?:of\s+)?['\"]?(\w+)['\"]?", re.I), lambda m: m.group(1)[::-1]),
]

def solve_string(prompt):
    for pattern, fn in _STRING_PATTERNS:
        m = pattern.search(prompt)
        if m and len(m.group(1)) > 1:
            return fn(m)
    return None

def solve_counting(prompt):
    lower = prompt.lower()
    count_type = None
    if re.search(r'\b(count|how many)\b', lower):
        if re.search(r'\bwords?\b', lower):
            count_type = 'words'
        elif re.search(r'\bvowels?\b', lower):
            count_type = 'vowels'
        elif re.search(r'\b(digits?|numbers?)\b', lower):
            count_type = 'digits'
        elif re.search(r'\b(letters?|characters?|chars?)\b', lower):
            count_type = 'letters'
        elif re.search(r'\bsentences?\b', lower):
            count_type = 'sentences'
    if not count_type:
        return None
    quote = re.search(r"['\"]([^'\"]+)['\"]", prompt)
    if not quote:
        return None
    text = quote.group(1)
    if count_type == 'words':
        return str(len(text.split()))
    if count_type == 'vowels':
        return str(sum(1 for c in text.lower() if c in 'aeiou'))
    if count_type == 'digits':
        return str(sum(1 for c in text if c.isdigit()))
    if count_type == 'letters':
        return str(sum(1 for c in text if c.isalpha()))
    if count_type == 'sentences':
        return str(len(re.findall(r'[.!?]+', text)))
    return None

_COMP_OPS = {
    'greater than': lambda a, b: a > b,
    'less than': lambda a, b: a < b,
    'equal to': lambda a, b: a == b,
    'greater than or equal to': lambda a, b: a >= b,
    'less than or equal to': lambda a, b: a <= b,
    'not equal to': lambda a, b: a != b,
}

def solve_true_false(prompt):
    text = prompt.strip()
    m = re.search(r'\b[Ii]s\s+(\d+(?:\.\d+)?)\s+(greater than|less than|equal to|greater than or equal to|less than or equal to|not equal to)\s+(\d+(?:\.\d+)?)', text)
    if m and m.group(2) in _COMP_OPS:
        a, b = float(m.group(1)), float(m.group(3))
        return str(_COMP_OPS[m.group(2)](a, b))
    m = re.search(r"\b[Ii]s\s+'([^']+)'\s+(in|not in)\s+'([^']+)'", text)
    if m:
        x, y, op = m.group(1), m.group(3), m.group(2)
        return str(op == 'in' and x in y or op == 'not in' and x not in y)
    m = re.search(r'\b[Ii]s\s+(\d+(?:\.\d+)?)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)', text)
    if m:
        n, a, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return str(a <= n <= b)
    return None

def solve_percentage(prompt):
    m = re.search(r'(?:what|what is|calculate|find)\s+(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)', prompt, re.I)
    if not m:
        m = re.search(r'(?:what|what is|calculate|find)\s+(\d+(?:\.\d+)?)\s+percent\s+of\s+(\d+(?:\.\d+)?)', prompt, re.I)
    if not m:
        m = re.search(r'^(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)', prompt, re.I)
    if m:
        p, v = float(m.group(1)), float(m.group(2))
        r = p / 100 * v
        return str(int(r) if r == int(r) else r)
    m = re.search(r'(?:what|what percent)\s+of\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)', prompt, re.I)
    if m:
        total, part = float(m.group(1)), float(m.group(2))
        if total == 0:
            return None
        r = part / total * 100
        return str(int(r) if r == int(r) else r)
    return None

def solve_date_math(prompt):
    text = prompt.strip()
    m = re.search(r'(\d+)\s+(days?|weeks?|months?)\s+(after|before|from)\s+(\d{4}-\d{2}-\d{2})', text, re.I)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        direction = 1 if m.group(3).lower() in ('after', 'from') else -1
        try:
            base = datetime.strptime(m.group(4), '%Y-%m-%d').date()
        except ValueError:
            return None
        if unit.startswith('day'):
            delta = timedelta(days=amount * direction)
        elif unit.startswith('week'):
            delta = timedelta(weeks=amount * direction)
        elif unit.startswith('month'):
            return None
        result = base + delta
        return result.isoformat()
    m = re.search(r'(?:how many|number of)\s+(days?|weeks?)\s+(?:between|from)\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2})', text, re.I)
    if m:
        try:
            d1 = datetime.strptime(m.group(2), '%Y-%m-%d').date()
            d2 = datetime.strptime(m.group(3), '%Y-%m-%d').date()
        except ValueError:
            return None
        diff = abs((d2 - d1).days)
        if m.group(1).lower().startswith('week'):
            return str(diff // 7)
        return str(diff)
    m = re.search(r'(?:how many|number of)\s+hours?\s+(?:between|from)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+(?:and|to)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)', text, re.I)
    if not m:
        m = re.search(r'(?:how many|number of)\s+hours?\s+(?:between|from)\s+(\d{1,2})\s*(am|pm)?\s+(?:and|to)\s+(\d{1,2})\s*(am|pm)?', text, re.I)
    if m:
        def _h2m(h, mi, ap):
            h, mi = int(h), int(mi) if mi else 0
            if ap:
                ap = ap.lower()
                if ap == 'pm' and h < 12: h += 12
                elif ap == 'am' and h == 12: h = 0
            return h * 60 + mi
        t1 = _h2m(m.group(1), m.group(2), m.group(3))
        t2 = _h2m(m.group(4), m.group(5), m.group(6))
        return str(abs(t2 - t1) // 60)
    return None

_UNIT_ALIAS = {
    'km': 'km', 'kilometer': 'km', 'kilometers': 'km',
    'mi': 'mi', 'mile': 'mi', 'miles': 'mi',
    'm': 'm', 'meter': 'm', 'meters': 'm',
    'ft': 'ft', 'foot': 'ft', 'feet': 'ft',
    'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
    'lb': 'lb', 'lbs': 'lb', 'pound': 'lb', 'pounds': 'lb',
    'l': 'l', 'liter': 'l', 'liters': 'l',
    'gal': 'gal', 'gallon': 'gal', 'gallons': 'gal',
}

_CONV_FACTORS = {
    ('km', 'mi'): 0.621371, ('mi', 'km'): 1.60934,
    ('m', 'ft'): 3.28084, ('ft', 'm'): 0.3048,
    ('kg', 'lb'): 2.20462, ('lb', 'kg'): 0.453592,
    ('l', 'gal'): 0.264172, ('gal', 'l'): 3.78541,
}

def solve_convert(prompt):
    text = prompt.strip()
    m = re.search(r'\bconvert\s+(\d+(?:\.\d+)?)\s*°?\s*(c|f|celsius|fahrenheit)\s+to\s+(c|f|celsius|fahrenheit)', text, re.I)
    if m:
        v = float(m.group(1))
        fr = m.group(2).lower()[0]
        to = m.group(3).lower()[0]
        if fr == 'c' and to == 'f':
            r = v * 9/5 + 32
        elif fr == 'f' and to == 'c':
            r = (v - 32) * 5/9
        else:
            return None
        r = round(r, 4)
        return str(int(r) if r == int(r) else r)
    m = re.search(r'\bconvert\s+(\d+(?:\.\d+)?)\s*(\w+)\s+to\s+(\w+)', text, re.I)
    if m:
        v = float(m.group(1))
        fr = _UNIT_ALIAS.get(m.group(2).lower())
        to = _UNIT_ALIAS.get(m.group(3).lower())
        if fr and to and (fr, to) in _CONV_FACTORS:
            r = v * _CONV_FACTORS[(fr, to)]
            r = round(r, 4)
            return str(int(r) if r == int(r) else r)
    return None

_ADD_W = re.compile(r'\b(plus|and|more|another|add|gain|earn|buy|find|receive|get|collect|increase|added|gained|earned|bought|found|received)\b')
_SUB_W = re.compile(r'\b(minus|less|fewer|away|lose|spend|give|sell|pay|remove|eat|take|left|remaining|use|uses|used|gave|spent|sold|paid|lost|took|removed)\b')
_MUL_W = re.compile(r'\b(times|multiplied by|double|triple|each|per)\b')

def solve_word_problem(prompt):
    lower = prompt.lower()
    if not re.search(r'\b(has?|have|had|there (?:is|are|was|were)|how (?:many|much|far)|left|total|altogether)\b', lower):
        return None
    if re.search(r'[+\-*/]', prompt):
        return None
    matches = list(re.finditer(r'\d+(?:\.\d+)?', lower))
    if len(matches) < 2:
        return None
    nums = [float(m.group()) for m in matches]
    result = nums[0]
    valid = False
    for i in range(len(nums) - 1):
        seg = lower[matches[i].end():matches[i+1].start()]
        if _MUL_W.search(seg):
            result *= nums[i+1]
            valid = True
        elif _SUB_W.search(seg):
            result -= nums[i+1]
            valid = True
        elif _ADD_W.search(seg) or re.search(r',\s*$|and\b', seg):
            result += nums[i+1]
            valid = True
        else:
            return None
    if not valid:
        return None
    if result == int(result):
        return str(int(result))
    return str(result)

_CODE_RESULT_RE = re.compile(r"(?:result|answer|output)\s*[=:]\s*(-?\d+(?:\.\d+)?|True|False|None|['\"][^'\"]+['\"])", re.I)
_CODE_RETURN_RE = re.compile(r"return\s+(-?\d+(?:\.\d+)?|True|False|None|['\"][^'\"]+['\"])", re.I)

def solve_code_direct(prompt):
    if not re.search(r"```|def |function |class ", prompt):
        return None
    for pattern in [_CODE_RESULT_RE, _CODE_RETURN_RE]:
        m = pattern.search(prompt)
        if m:
            return m.group(1).strip().strip("'\"")
    return None

_LOGIC_COMP_RE = re.compile(r"\b(and|or|not)\b", re.I)
_LOGIC_EVAL_RE = re.compile(
    r"((?:True|False)\s+(?:and|or)\s+(?:True|False)|"
    r"not\s+\((?:True|False)\s+(?:and|or)\s+(?:True|False)\)|"
    r"not\s+(?:True|False))", re.I)

def solve_logical_basic(prompt):
    if not _LOGIC_COMP_RE.search(prompt):
        return None
    text = prompt.strip()
    for pattern in [_LOGIC_EVAL_RE]:
        for m in pattern.finditer(text):
            expr = m.group(0).strip()
            try:
                result = eval(expr, {"__builtins__": {}}, {"True": True, "False": False})
                return str(result)
            except Exception:
                continue
    m = re.search(r"\b(True|False)\b", text)
    if m and _LOGIC_COMP_RE.search(text):
        val = m.group(0)
        try:
            result = eval("not " + val if "not" in text.lower() else val,
                          {"__builtins__": {}}, {"True": True, "False": False})
            return str(result)
        except Exception:
            return m.group(0)
    return None

def solve_answer_in_prompt(prompt):
    m = re.search(r'(?:answer|result)(?:\s+is|\s*:\s*|=)\s*(.+?)(?:\.\s*$|$)', prompt, re.I)
    if m:
        val = _FACTUAL_STRIP.sub('', m.group(1)).strip()
        if val:
            return val
    return None

_FACTUAL_STRIP = re.compile(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$")

def solve_factual_echo(prompt):
    sents = re.split(r'(?<=[.!?])\s+', prompt.strip())
    has_q = any('?' in s for s in sents)
    if not has_q:
        return None
    non_q = [s for s in sents if '?' not in s
             and not s.strip().lower().startswith(('classify','extract','summarize','evaluate','calculate','count','uppercase','reverse','fix','write','output',"'",'"'))]
    non_q = [s for s in non_q if re.match(r'^(There|The|A|An|[A-Z][a-z]+)', s.strip())]
    if not non_q:
        return None
    first = non_q[0].strip().rstrip('.')
    patterns = [
        r'\bthere are\s+(\d+)(?:\s+\w+)?$',
        r'\bthere are\s+(.+?)$',
        r'\bhas the (?:formula|chemical formula|name)\s+(.+?)$',
        r'\b(?:was|were)\s+(?:\w+\s+){0,3}by\s+(.+?)$',
        r'\b(?:is|was|are|were)\s+(?!\w+\s+by\b)(.+?)$',
    ]
    for pat in patterns:
        m = re.search(pat, first, re.I)
        if m:
            val = _FACTUAL_STRIP.sub('', m.group(1)).strip()
            if val:
                return val
    return None

DETERMINISTIC_SOLVERS = [
    ("answer_in_prompt", solve_answer_in_prompt),
    ("factual_echo", solve_factual_echo),
    ("count", solve_counting),
    ("true_false", solve_true_false),
    ("percentage", solve_percentage),
    ("date_math", solve_date_math),
    ("convert", solve_convert),
    ("word_problem", solve_word_problem),
    ("math", solve_math),
    ("code", solve_code_direct),
    ("logical", solve_logical_basic),
    ("json", solve_json),
    ("extract", solve_extract),
    ("sentiment", solve_sentiment),
    ("string", solve_string),
]

def try_deterministic(prompt):
    for name, solver in DETERMINISTIC_SOLVERS:
        try:
            answer = solver(prompt)
            if answer:
                return answer
        except Exception:
            continue
    return None
