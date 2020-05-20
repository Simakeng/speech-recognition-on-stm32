from pypinyin import pinyin, lazy_pinyin, Style


def get_tokens():
    tokens = [
        'a', 'a1', 'a2', 'a3', 'a4',
        'o', 'o1', 'o2', 'o3', 'o4',
        'e', 'e1', 'e2', 'e3', 'e4',
        'i', 'i1', 'i2', 'i3', 'i4',
        'u', 'u1', 'u2', 'u3', 'u4',
        'v', 'v1', 'v2', 'v3', 'v4',
        'n2',
    ]
    for ch in 'bpmfdtnlgkhjqxzcsryw':
        tokens.append(ch)
    tokens.append('zh')
    tokens.append('ch')
    tokens.append('sh')
    tokens.append('ng')

    return tokens


cleas = [' ', '，', '。', '？', '！', ',', '.', '!', '?']


def tokenize(s):
    for c in cleas:
        s = s.replace(c, '')
    tokens = get_tokens()
    letter_pinyins = pinyin(s, style=Style.TONE2, heteronym=False)
    result = []
    for p in letter_pinyins:
        p = p[0]
        while len(p) != 0:
            if (len(p) >= 2) and(p[:2] in tokens):
                result.append(p[:2])
                p = p[2:]
            elif p[0] in tokens:
                result.append(p[0])
                p = p[1:]
            else:
                raise Exception('illegal charater')
    return result


def index_token(tokens):
    lt = get_tokens()
    return [lt.index(tk) for tk in tokens]
