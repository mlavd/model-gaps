from collections import defaultdict
from .generator import *
from .translators import BasicCTranslator
import pytest

def do_50_50_test(gen, max_tests=10_000):
    # This tests to make sure each operator is 50/50
    is_true = 0

    for _ in range(max_tests):
        try:
            node = gen.generate()
            is_true += int(node.evaluate())
        except ZeroDivisionError:
            pass

    assert pytest.approx(is_true / max_tests, 0.1) == 0.5

class TestTask1Generator():
    def setup_method(self):
        self.gen = Task1Generator()
        self.trans = BasicCTranslator()

    def test_random(self):
        random.seed(41)
        results = [
            ('2485 <= 886', False),
            ('-4560 == -4560', True),
            ('-712 > 8122', False),
            ('2567 < 8888', True),
            ('-1839 >= -9400', True),
            ('-4920 != -5091', True),
            ('-1562 < 9129', True),
            ('-6074 < 9259', True),
            ('4175 != -765', True),
            ('1803 <= 5570', True),
            ('8194 < -9393', False),
            ('-6767 != -6767', False),
            ('-4176 > -8246', True),
            ('413 == -2617', False),
            ('8932 >= -4034', True),
            ('-8120 >= -124', False),
            ('-5219 != -6504', True),
            ('-9296 >= 6163', False),
            ('-2451 < -9291', False),
            ('-5578 >= -7449', True),
        ]

        for exp_str, exp_eval in results:
            result = self.gen.generate()
            assert self.trans.translate(result) == exp_str
            assert result.evaluate({}) == exp_eval

    def test_is_50_50(self):
        do_50_50_test(self.gen)


class TestTask2Generator():
    def setup_method(self):
        self.gen = Task2Generator()
        self.trans = BasicCTranslator()

    def test_random(self):
        random.seed(42)
        results = [
            ('u = 9309;\nd = -885;\nu >= d', True),
            ('e = -8499;\ni = -7721;\ni < e', False),
            ('d = 626;\nm = -8168;\nd > m', True),
            ('i = -6408;\nc = -4992;\ni < c', True),
            ('t = 7674;\nm = -4707;\nm >= t', False),
            ('l = 3074;\nd = 214;\nl >= d', True),
            ('h = 8711;\nz = 8881;\nz >= h', True),
            ('d = -1815;\ng = 5159;\ng < d', False),
            ('r = -2867;\nu = -278;\nu > r', True),
            ('b = 295;\ns = -2179;\ns <= b', True),
            ('k = -5600;\no = -6606;\nk < o', False),
            ('j = -9684;\ni = 8379;\ni <= j', False),
            ('s = -9188;\nl = 3510;\nl >= s', True),
            ('f = 851;\ni = 7570;\ni > f', True),
            ('s = -5917;\ni = -160;\ni > s', True),
            ('w = -7222;\nj = -701;\nj <= w', False),
            ('g = 8265;\nh = -8355;\nh >= g', False),
            ('z = -7462;\no = -638;\nz > o', False),
            ('r = 1526;\nc = -216;\nc <= r', True),
            ('h = -1476;\ni = 2422;\ni >= h', True),
        ]

        for exp_str, exp_eval in results:
            result = self.gen.generate()
            assert self.trans.translate(result) == exp_str
            assert result.evaluate() == exp_eval

    def test_is_50_50(self):
        do_50_50_test(self.gen)


class TestTask3Generator():
    def setup_method(self):
        self.gen = Task3Generator()
        self.trans = BasicCTranslator()

    def test_random(self):
        random.seed(42)
        results = [
            ('u = 9309;\nd = -885;\nu = u * -2945;\nu = -6961 * u;\nd = 9782 + d;\nu >= d', True),
            ('o = -1252;\nx = -3135;\no = o % -1155;\nx = x + 626;\nx = x / -8949;\no = -7832 % o;\nx <= o', False),
            ('w = -6408;\nk = -4992;\nk = 7338 % k;\nw = w % -6247;\nk = 1146 / k;\nw = 4867 * w;\nw < k', True),
            ('y = 8591;\nq = -2108;\nq = q % -5793;\nq = q % -1315;\ny = y - -3060;\ny = y * 3074;\ny = 6959 - y;\nq >= y', True),
            ('h = 8711;\nc = 8881;\nh = h * -6824;\nc = 3470 + c;\nc >= h', True),
            ('v = -6944;\nu = -2255;\nu = -8079 + u;\nu = u / -1310;\nv >= u', False),
            ('n = -7755;\nj = -3911;\nj = j % 8970;\nj = j % 3737;\nn = -1456 * n;\nj = -1297 * j;\nn = n + 360;\nj >= n', False),
            ('o = -5600;\na = -6606;\no = 9819 * o;\na = a * -1350;\na = a * -6976;\no < a', False),
            ('b = 1724;\na = -3117;\nb = b - -2243;\na = -9188 * a;\na = -1869 + a;\na = a - -8732;\nb = 5083 - b;\nb > a', False),
            ('a = 278;\nh = 1327;\nh = 4288 % h;\nh = 1934 * h;\na = a * -5917;\na >= h', False),
            ('q = 9004;\nj = 9879;\nj = j * -4440;\nj = j - -6940;\nj >= q', False),
            ('j = 6972;\nv = -4237;\nv = v % 5225;\nj = j % 4502;\nj = 7952 - j;\nj = 4747 - j;\nj <= v', True),
            ('i = 842;\nu = 3356;\ni = i + 9139;\ni = 5629 * i;\ni = i / 2779;\nu = 5998 / u;\ni < u', False),
            ('p = 4899;\na = 710;\np = 5411 % p;\np = 1467 + p;\np = p - -8984;\na > p', False),
            ('a = 2997;\nt = -3502;\na = a * 8544;\nt = 2142 * t;\na = a + 3764;\nt = t / 1868;\na = 4270 % a;\na < t', False),
            ('i = -1535;\nu = 1626;\nu = -7195 / u;\ni = i / 8192;\ni = i + 4685;\ni = -2739 - i;\ni = 9056 % i;\ni > u', False),
            ('l = -3640;\nq = -6706;\nq = 7502 + q;\nq = 8132 - q;\nl = l % -6639;\nl > q', False),
            ('j = 8286;\np = 3637;\np = p + 6;\nj = j * -3192;\nj = -4854 % j;\nj >= p', False),
            ('c = -1296;\nf = -5340;\nf = 4390 * f;\nf = f / -1080;\nf = f / -7365;\nf = 9782 + f;\nc = c + 8848;\nc <= f', True),
            ('v = -556;\ny = 6235;\nv = 593 / v;\ny = 7779 + y;\ny = 6988 / y;\nv >= y', False),
        ]

        for exp_str, exp_eval in results:
            result = self.gen.generate()
            assert self.trans.translate(result) == exp_str
            assert result.evaluate() == exp_eval

    def test_is_50_50(self):
        do_50_50_test(self.gen)


class TestTask4Generator():
    def setup_method(self):
        self.gen = Task4Generator()
        self.trans = BasicCTranslator()

    def test_random(self):
        random.seed(42)
        results = [
            ('u = 9309;\nd = -885;\nif (u > -895) {\n\tu = -6961 * u;\n} else {\n\td = 9782 + d;\n}\nu >= d', False),
            ('o = -1252;\nx = -3135;\nif (o <= 5147) {\n\to = -8168 + o;\n} else {\n\to = 3145 - o;\n}\nx <= o', False),
            ('s = -6408;\nw = -4992;\nif (2504 > w) {\n\ts = s % -6247;\n} else {\n\tw = 1146 / w;\n}\ns < w', False),
            ('o = -2108;\nf = -8102;\nif (f >= 7455) {\n\tf = f % -1315;\n} else {\n\to = o - -3060;\n}\nf >= o', False),
            ('j = -7679;\nw = 827;\nif (w <= j) {\n\tj = 3338 + j;\n} else {\n\tj = 1609 / j;\n}\nw >= j', True),
            ('b = 7713;\nx = -8343;\nif (b <= 3317) {\n\tb = -8079 + b;\n} else {\n\tb = b / -1310;\n}\nx <= b', True),
            ('n = -7755;\nj = -3911;\nif (j < 8667) {\n\tn = n * 9125;\n} else {\n\tn = n * -3307;\n}\nj >= n', True),
            ('i = 4358;\nh = -4831;\nif (i < h) {\n\th = -1076 - h;\n} else {\n\ti = i * -3329;\n}\ni > h', False),
            ('p = -8820;\nq = 7877;\nif (p < -8630) {\n\tp = p + -3117;\n} else {\n\tp = p / 8346;\n}\np >= q', False),
            ('x = 0;\nt = 1457;\nif (x < 3056) {\n\tx = x / 1506;\n} else {\n\tx = -9096 * x;\n}\nt < x', False),
            ('f = -5917;\ns = -160;\nif (-312 <= f) {\n\ts = s - 2423;\n} else {\n\ts = s + 3306;\n}\ns > f', True),
            ('j = -9200;\nz = -8486;\nif (3580 >= j) {\n\tz = 3095 + z;\n} else {\n\tj = 3931 % j;\n}\nj < z', True),
            ('b = 5879;\no = 7081;\nif (o > b) {\n\tb = 7699 - b;\n} else {\n\to = o - 2551;\n}\no >= b', True),
            ('n = -2810;\nc = 5998;\nif (n <= 2742) {\n\tn = n % -5818;\n} else {\n\tc = c + 2911;\n}\nn >= c', False),
            ('u = -8681;\nc = -7752;\nif (c >= -5007) {\n\tc = c / 8480;\n} else {\n\tu = 2087 % u;\n}\nc <= u', True),
            ('x = 1377;\nw = 7456;\nif (w < -9586) {\n\tw = w / 1868;\n} else {\n\tx = 4270 % x;\n}\nw < x', False),
            ('i = -1535;\nu = 1626;\nif (u <= 6928) {\n\ti = i - 6009;\n} else {\n\ti = 6083 + i;\n}\ni > u', False),
            ('c = 7559;\nj = -3934;\nif (9319 < j) {\n\tc = -293 - c;\n} else {\n\tj = j - -9537;\n}\nc >= j', True),
            ('b = -5112;\ni = -5029;\nif (b >= i) {\n\tb = 4507 % b;\n} else {\n\tb = b + 6;\n}\nb <= i', True),
            ('x = -2352;\nt = -7668;\nif (t > 7844) {\n\tt = -7661 * t;\n} else {\n\tx = 4390 * x;\n}\nx >= t', False),
        ]

        for exp_str, exp_eval in results:
            result = self.gen.generate()
            assert self.trans.translate(result) == exp_str
            assert result.evaluate() == exp_eval

    def test_is_50_50(self):
        do_50_50_test(self.gen)
