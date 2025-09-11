# benchmark.py
from dataclasses import dataclass
from typing import Tuple, Union

@dataclass
class BenchmarkFunction:
    name: str
    range: Tuple[Union[float, str], Union[float, str]]
    dimension: int
    global_minima: Union[float, str]  # can be float or string
    type: str  # "unimodal" or "multimodal"

benchmark_functions = [
    BenchmarkFunction("ackley", (-32, 32), 30, 0, "multimodal"),
    BenchmarkFunction("ackleyn2", (-32, 32), 2, -200, "unimodal"),
    BenchmarkFunction("ackleyn3", (-32, 32), 2, -195.629028238419, "multimodal"),
    BenchmarkFunction("ackleyn4", (-35, 35), 30, -4.590101633799122, "multimodal"),
    BenchmarkFunction("adjiman", (-1, 2), 2, -2.02181, "multimodal"),
    BenchmarkFunction("alpinen1", (0, 10), 30, 0, "multimodal"),
    BenchmarkFunction("alpinen2", (0, 10), 30, "2.83082553*10^13", "multimodal"),
    BenchmarkFunction("bartelsconn", (-500, 500), 2, 1, "multimodal"),
    BenchmarkFunction("beale", (-4.5, 4.5), 2, 0, "multimodal"),
    BenchmarkFunction("bird", (-6.28,6.28), 2, -106.764537, "multimodal"),
    BenchmarkFunction("bohachevskyn1", (-100, 100), 2, 0, "unimodal"),
    BenchmarkFunction("bohachevskyn2", (-100, 100), 2, 0, "multimodal"),
    BenchmarkFunction("booth", (-10, 10), 2, 0, "unimodal"),
    BenchmarkFunction("brent", (-20, 0), 2, "e^-200", "unimodal"),
    BenchmarkFunction("brown", (-1, 4), 30, 0, "unimodal"),
    BenchmarkFunction("bukinn6", (-13, 3), 2, 0, "multimodal"),
    BenchmarkFunction("carromtable", (-10, 10), 2, -24.15681551650653, "multimodal"),
    BenchmarkFunction("crossintray", (-10, 10), 2, -2.06261218, "multimodal"),
    BenchmarkFunction("deckkersaarts", (-20, 20), 2, -24771.09375, "multimodal"),
    BenchmarkFunction("dropwave", (-5.2, 5.2), 2, -1, "unimodal"),
    BenchmarkFunction("easom", (-100, 100), 2, -1, "multimodal"),
    BenchmarkFunction("eggcrate", (-5, 5), 2, 0, "multimodal"),
    BenchmarkFunction("elattar", (-500, 500), 2, 0, "multimodal"),
    BenchmarkFunction("exponential", (-1, 1), 2, -1, "unimodal"),
    BenchmarkFunction("forrester", (-0.5, 2.5), 1, 6.0207, "multimodal"),
    BenchmarkFunction("goldsteinprice", (-2, 2), 2, 3, "multimodal"),
    BenchmarkFunction("gramacylee", (-0.5, 2.5), 1, -0.869011134989500, "multimodal"),
    BenchmarkFunction("griewank", (-600, 600), 30, 0, "unimodal"),
    BenchmarkFunction("happycat", (-2, 2), 2, 30, "multimodal"),
    BenchmarkFunction("himmelblau", (-6, 6), 2, 0, "multimodal"),
    BenchmarkFunction("holdertable", (-10, 10), 2, -19.2085, "multimodal"),
    BenchmarkFunction("keane", (0, 10), 2, 0.673667521146855, "multimodal"),
    BenchmarkFunction("leon", (0, 10), 2, 0, "unimodal"),
    BenchmarkFunction("levin13", (-10, 10), 2, 0, "multimodal"),
    BenchmarkFunction("matyas", (-10, 10), 2, 0, "unimodal"),
    BenchmarkFunction("mccormick", (-3, 3), 2, -1.9133, "multimodal"),
    BenchmarkFunction("periodic", (-10, 10), 30, 0.9, "multimodal"),
    BenchmarkFunction("powellsum", (-1, 1), 30, 0, "unimodal"),
    BenchmarkFunction("qing", (-500, 500), 30, 0, "multimodal"),
    BenchmarkFunction("quartic", (-1.28, 1.28), 30, 0, "multimodal"),
    BenchmarkFunction("rastrigin", (-5.12, 5.12), 30, 0, "multimodal"),
    BenchmarkFunction("ridge", (-5, 5), 30, -5, "unimodal"),
    BenchmarkFunction("rosenbrock", (-5, 10), 30, 0, "multimodal"),
    BenchmarkFunction("salomon", (-100, 100), 30, 0, "multimodal"),
    BenchmarkFunction("schaffern1", (-100, 100), 2, 0, "unimodal"),
    BenchmarkFunction("schaffern2", (-100, 100), 2, 0, "unimodal"),
    BenchmarkFunction("schaffern3", (-100, 100), 2, 0.00156685, "unimodal"),
    BenchmarkFunction("schaffern4", (-100, 100), 2, 0.292579, "unimodal"),
    BenchmarkFunction("schwefel220", (-100, 100), 30, 0, "unimodal"),
    BenchmarkFunction("schwefel221", (-100, 100), 30, 0, "unimodal"),
    BenchmarkFunction("schwefel222", (-100, 100), 30, 0, "unimodal"),
    BenchmarkFunction("schwefel223", (-100, 100), 30, 0, "unimodal"),
    BenchmarkFunction("schwefel", (-500, 500), 30, 0, "multimodal"),
    BenchmarkFunction("shubertn3", (-10, 10), 30, -29.6733337, "multimodal"),
    BenchmarkFunction("shubertn4", (-10, 10), 30, -25.740858, "multimodal"),
    BenchmarkFunction("shubert", (-10, 10), 30, -186.7309, "multimodal"),
    BenchmarkFunction("sphere", (-5.12, -5.12), 30, 0, "unimodal"),
    BenchmarkFunction("styblinskitank", (-5, 5), 30, 39.16599, "multimodal"),
    BenchmarkFunction("sumsquares", (-10, 10), 30, 0, "unimodal"),
    BenchmarkFunction("threehumpcamel", (-5, 5), 2, 0, "unimodal"),
    BenchmarkFunction("trid", (-900, 900), 30, -4930, "unimodal"),
    BenchmarkFunction("wolfe", (0, 2), 3, 0, "multimodal"),
    BenchmarkFunction("xinsheyangn1", (-5, 5), 30, 0, "multimodal"),
    BenchmarkFunction("xinsheyangn2", (-6.28,6.28), 30, 0, "multimodal"),
    BenchmarkFunction("xinsheyangn3", (-6.28,6.28), 30, -1, "unimodal"),
    BenchmarkFunction("xinsheyangn4", (-10, 10), 30, -1, "multimodal"),
    BenchmarkFunction("zakharov", (-5, 10), 30, 0, "unimodal")
]

# '', '', '', '', '', 'yaoliun4', 'yaoliun9', '', 'zerosum', 'zettel', 'zimmerman', 'zirilli'
# # 'crossintray', 'crownedcross', 'csendes', 'cubefcn', 'debn1', '',
#  'bukinn4', '', '', 'chichinadze', 'cigar', 'cosinemixture',
#, 'amgm', 'annotations', 'braninn1', 'braninn2', '', '', 'bukinn2',