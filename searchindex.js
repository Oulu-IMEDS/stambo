Search.setIndex({"docnames": ["Classification", "Regression", "Two_sample_test", "index", "metrics", "stambo"], "filenames": ["Classification.nblink", "Regression.nblink", "Two_sample_test.nblink", "index.rst", "metrics.rst", "stambo.rst"], "titles": ["Comparing two classification models using <code class=\"docutils literal notranslate\"><span class=\"pre\">stambo</span></code>", "Comparing two regression models using <code class=\"docutils literal notranslate\"><span class=\"pre\">stambo</span></code>", "Comparing two samples using <code class=\"docutils literal notranslate\"><span class=\"pre\">stambo</span></code>", "Welcome to stambo\u2019s documentation!", "Metrics", "Main functionality"], "terms": {"v1": [0, 1, 2], "1": [0, 1, 2, 5], "aleksei": [0, 1, 2], "tiulpin": [0, 1, 2], "phd": [0, 1, 2], "2024": [0, 1, 2], "thi": [0, 1, 2], "notebook": [0, 1], "show": [0, 1, 2, 5], "an": [0, 1, 5], "end": [0, 1, 2], "exampl": [0, 1, 2], "how": [0, 1, 2], "one": [0, 1, 5], "can": [0, 1], "take": [0, 1, 4], "machin": [0, 1, 2, 5], "learn": [0, 1, 2, 5], "conduct": [0, 1, 2], "assess": [0, 1, 2], "whether": [0, 1, 2, 5], "ar": [0, 1, 2, 5], "differ": [0, 1, 5], "we": [0, 1, 2, 5], "first": [0, 2], "set": [0, 5], "classic": 0, "basic": 0, "from": [0, 1, 5], "sklearn": [0, 1], "At": 0, "tutori": 0, "gener": [0, 5], "latex": [0, 1, 5], "report": [0, 1, 5], "implement": 0, "custom": [0, 5], "numpi": [0, 1, 2], "np": [0, 1, 2, 5], "neighbor": [0, 1], "kneighborsclassifi": 0, "linear_model": [0, 1], "logisticregress": 0, "load_breast_canc": 0, "model_select": [0, 1], "train_test_split": [0, 1], "preprocess": [0, 1], "standardscal": [0, 1], "roc_auc_scor": 0, "average_precision_scor": 0, "seed": [0, 1, 2, 5], "2": [0, 1, 2, 5], "x": [0, 1, 2, 5], "y": [0, 1, 5], "return_x_i": [0, 1], "true": [0, 1, 5], "xtr": [0, 1], "xte": [0, 1], "ytr": [0, 1], "yte": [0, 1], "test_siz": [0, 1], "0": [0, 1, 2, 5], "5": [0, 1, 2], "random_st": [0, 1], "stratifi": 0, "scaler": [0, 1], "fit": [0, 1], "transform": [0, 1], "knn": [0, 1], "logist": [0, 1], "regress": [0, 2, 3], "here": [0, 1, 5], "see": [0, 1], "outperform": [0, 1], "3": [0, 1, 2], "n_neighbor": [0, 1], "preds_knn": [0, 1], "predict_proba": 0, "c": 0, "1e": 0, "42": [0, 1], "preds_lr": [0, 1], "auc_knn": 0, "auc_lr": 0, "print": [0, 1, 2], "f": [0, 1], "auc": [0, 4], "4f": [0, 1], "lr": [0, 1], "9722": 0, "9918": 0, "As": [0, 1], "state": [0, 1, 5], "document": [0, 1], "routin": [0, 1], "return": [0, 1, 5], "dict": [0, 1, 5], "tupl": [0, 1, 5], "The": [0, 1, 2, 4, 5], "kei": [0, 1], "tag": [0, 1], "valu": [0, 1, 2, 5], "store": [0, 1], "data": [0, 1, 5], "follow": [0, 1, 5], "format": [0, 1, 5], "p": [0, 1, 2, 5], "h_0": [0, 1, 5], "model_1": [0, 1], "model_2": [0, 1], "empir": [0, 1, 5], "ci": [0, 1, 5], "low": [0, 1, 5], "high": [0, 1, 5], "If": [0, 1], "you": [0, 1, 5], "launch": [0, 1], "code": [0, 1], "binder": [0, 1], "decreas": [0, 1], "number": [0, 1, 5], "bootstrap": [0, 1, 2, 5], "iter": [0, 1, 5], "10000": [0, 1, 2, 5], "default": [0, 1, 5], "4": [0, 2], "testing_result": [0, 1], "compare_model": [0, 1, 3, 5], "rocauc": [0, 3, 4], "ap": [0, 3, 4], "qkappa": [0, 3, 4], "bacc": [0, 3, 4], "mcc": [0, 3, 4], "100": [0, 1, 2], "00": [0, 1, 2], "17": [0, 1], "lt": [0, 1, 2], "576": 0, "63it": 0, "": [0, 1, 2, 4], "want": [0, 1, 2], "visual": [0, 1], "result": [0, 1], "thei": [0, 1, 5], "avail": [0, 1, 5], "have": [0, 1, 2, 5], "describ": [0, 1], "abov": [0, 1], "39": [0, 1], "0165983401659834": 0, "9721724465057446": 0, "9488642065294073": 0, "991257028580809": 0, "9917782228312428": 0, "9796223194281446": 0, "9991088801592725": 0, "018598140185981403": 0, "9699899675866734": 0, "9431022140624091": 0, "9908460876062002": 0, "9940360662959732": 0, "9843501977589723": 0, "9994975413122237": 0, "30716928307169283": 0, "8936283657691282": 0, "8359946182323871": 0, "9445911828990862": 0, "8844563366577475": 0, "8238384670856083": 0, "9383926972823168": 0, "17638236176382363": 0, "9416570043217034": 0, "910371840928929": 0, "9699502854178561": 0, "9311689680615579": 0, "8970587113050348": 0, "9627659574468085": 0, "4393560643935606": 0, "8945584078905953": 0, "8380851036550386": 0, "9448910810319505": 0, "8889244497451684": 0, "8329384161226888": 0, "9399414434378707": 0, "most": [0, 1], "commonli": [0, 1], "though": [0, 1], "them": [0, 1, 5], "paper": [0, 1], "present": [0, 1], "For": [0, 1], "function": [0, 1, 3], "to_latex": [0, 1, 2, 3, 5], "get": [0, 1, 2], "cut": [0, 1, 5], "past": [0, 1, 5], "tabular": [0, 1, 2, 5], "To": [0, 1], "need": [0, 1], "forget": [0, 1], "booktab": [0, 1, 2], "6": 0, "m1_name": [0, 1, 2, 5], "m2_name": [0, 1, 2, 5], "usepackag": [0, 1, 2], "do": [0, 1, 2], "begin": [0, 1, 2], "llllll": 0, "toprul": [0, 1, 2], "textbf": [0, 1, 2], "amp": [0, 1, 2], "midrul": [0, 1, 2], "97": 0, "95": 0, "99": [0, 2], "94": 0, "89": [0, 1], "84": 0, "91": 0, "98": 0, "88": 0, "82": 0, "93": 0, "90": 0, "96": 0, "83": 0, "02": [0, 2], "31": 0, "18": [0, 1], "44": [0, 1], "bottomrul": [0, 1, 2], "sometim": 0, "i": [0, 1, 2, 5], "enough": 0, "mai": [0, 2], "some": 0, "addit": 0, "let": 0, "u": 0, "defin": [0, 4, 5], "f2": 0, "score": [0, 2, 4], "7": [0, 2], "fbeta_scor": 0, "functool": 0, "partial": 0, "8": 0, "class": [0, 4], "f2score": 0, "def": 0, "__init__": 0, "self": 0, "none": [0, 5], "beta": 0, "int_input": [0, 4], "__str__": 0, "str": [0, 5], "9": 0, "12": 0, "797": 0, "50it": 0, "10": 0, "llll": 0, "11": 0, "18198180181981802": 0, "9711431742508325": 0, "9503694283719967": 0, "9875846501128666": 0, "9801762114537445": 0, "9665621734587252": 0, "9904153354632587": 0, "kneighborsregressor": 1, "linearregress": 1, "load_diabet": 1, "metric": [1, 2, 3, 5], "mean_absolute_error": 1, "mean_squared_error": 1, "14": 1, "16": 1, "predict": [1, 4, 5], "mae_knn": 1, "mae_lr": 1, "mae": [1, 3, 4], "51": 1, "2489": 1, "3217": 1, "note": [1, 2, 5], "error": [1, 4], "which": 1, "mean": [1, 2, 4], "lower": 1, "better": [1, 5], "contrari": 1, "classif": [1, 2, 3], "therefor": 1, "actual": 1, "ask": 1, "question": 1, "ha": 1, "larger": [1, 5], "than": [1, 2, 5], "linear": 1, "so": 1, "improv": 1, "mse": [1, 3, 4], "03": 1, "2571": 1, "56it": 1, "0007999200079992001": 1, "321658137655405": 1, "40": 1, "17458217663324": 1, "48": 1, "62883263591205": 1, "248868778280546": 1, "46": 1, "417496229260934": 1, "56": 1, "1538838612368": 1, "0008999100089991": 1, "3020": 1, "4335055268534": 1, "2508": 1, "771399983571": 1, "3583": 1, "767801911658": 1, "3978": 1, "893916540975": 1, "3293": 1, "007629462041": 1, "4723": 1, "732478632479": 1, "19": 1, "lll": 1, "32": 1, "63": 1, "43": 1, "77": 1, "25": 1, "15": 1, "01": 1, "73": [1, 2], "There": 2, "mani": 2, "case": 2, "when": [2, 5], "develop": 2, "model": [2, 3, 5], "other": [2, 5], "comput": [2, 5], "per": [2, 5], "datapoint": 2, "find": 2, "often": 2, "just": [2, 5], "measur": 2, "allow": 2, "easili": 2, "too": 2, "simpl": 2, "test": [2, 5], "synthet": 2, "simpli": 2, "gaussian": 2, "second": 2, "greater": 2, "random": [2, 5], "n_sampl": 2, "sample_1": [2, 5], "randn": 2, "sample_2": [2, 5], "come": 2, "doe": [2, 5], "requir": 2, "statist": [2, 5], "choic": 2, "subclass": 2, "re": 2, "two_sample_test": [2, 3, 5], "lambda": 2, "82516": 2, "31it": 2, "ll": 2, "53": 2, "34": 2, "78": 2, "58": 2, "main": 3, "f1score": [3, 4], "compar": [3, 5], "two": [3, 4, 5], "us": [3, 5], "sampl": [3, 5], "stambo": [4, 5], "sourc": [4, 5], "base": 4, "averag": 4, "precis": 4, "binari": 4, "classifi": 4, "balanc": 4, "accuraci": 4, "f1": 4, "absolut": 4, "matthew": 4, "correl": 4, "coeffici": 4, "squar": 4, "callabl": [4, 5], "bool": [4, 5], "fals": [4, 5], "object": 4, "A": [4, 5], "wrapper": 4, "ground": [4, 5], "truth": [4, 5], "label": [4, 5], "argument": 4, "cohen": 4, "kappa": 4, "quadrat": 4, "roc": 4, "current": 5, "tail": 5, "anoth": 5, "comapr": 5, "evalu": 5, "y_test": 5, "ndarrai": 5, "ani": 5, "dtype": 5, "int64": 5, "float64": 5, "preds_1": 5, "preds_2": 5, "alpha": 5, "float": 5, "05": 5, "two_tail": 5, "n_bootstrap": 5, "int": 5, "silent": 5, "f_1": 5, "yield": 5, "vector": 5, "hat": 5, "y_": 5, "hypothesi": 5, "e": 5, "null": 5, "altern": 5, "hypothes": 5, "m": 5, "gt": 5, "h_1": 5, "where": 5, "respect": 5, "Such": 5, "kind": 5, "perform": 5, "everi": 5, "specifi": 5, "while": 5, "should": 5, "care": 5, "about": 5, "its": 5, "interpret": 5, "probabl": 5, "observ": 5, "least": 5, "extrem": 5, "obtain": 5, "assum": 5, "math": 5, "That": 5, "what": 5, "being": 5, "given": 5, "would": 5, "same": 5, "beyond": 5, "also": 5, "confid": 5, "interv": 5, "_": 5, "paramet": 5, "union": 5, "npt": 5, "call": 5, "user": 5, "either": 5, "librari": 5, "add": 5, "instanc": 5, "option": 5, "signific": 5, "level": 5, "execut": 5, "progress": 5, "bar": 5, "dictionari": 5, "contain": 5, "expect": 5, "output": 5, "entri": 5, "type": 5, "m1": 5, "m2": 5, "convert": 5, "tabl": 5, "conveni": 5, "view": 5, "name": 5, "assign": 5, "row": 5, "environ": 5, "predsamplewrapp": 5, "own": 5, "independ": 5, "thu": 5, "treat": 5, "_description_": 5}, "objects": {"": [[5, 0, 0, "-", "stambo"]], "stambo": [[5, 1, 1, "", "compare_models"], [4, 0, 0, "-", "metrics"], [5, 1, 1, "", "to_latex"], [5, 1, 1, "", "two_sample_test"]], "stambo.metrics": [[4, 2, 1, "", "AP"], [4, 2, 1, "", "BACC"], [4, 2, 1, "", "F1Score"], [4, 2, 1, "", "MAE"], [4, 2, 1, "", "MCC"], [4, 2, 1, "", "MSE"], [4, 2, 1, "", "Metric"], [4, 2, 1, "", "QKappa"], [4, 2, 1, "", "ROCAUC"]]}, "objtypes": {"0": "py:module", "1": "py:function", "2": "py:class"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"], "2": ["py", "class", "Python class"]}, "titleterms": {"compar": [0, 1, 2], "two": [0, 1, 2], "classif": 0, "model": [0, 1], "us": [0, 1, 2], "stambo": [0, 1, 2, 3], "import": [0, 1, 2], "necessari": [0, 1], "librari": [0, 1, 2], "load": [0, 1], "uci": 0, "breast": 0, "cancer": 0, "dataset": [0, 1], "creat": [0, 1], "train": [0, 1], "test": [0, 1], "split": [0, 1], "statist": [0, 1], "own": 0, "metric": [0, 4], "regress": 1, "diabet": 1, "sampl": 2, "data": 2, "gener": 2, "comparison": 2, "latex": 2, "report": 2, "welcom": 3, "": 3, "document": 3, "exampl": 3, "main": 5, "function": 5}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.viewcode": 1, "nbsphinx": 4, "sphinx": 60}, "alltitles": {"Comparing two classification models using stambo": [[0, "Comparing-two-classification-models-using-stambo"]], "Import of necessary libraries": [[0, "Import-of-necessary-libraries"], [1, "Import-of-necessary-libraries"]], "Loading the UCI breast cancer dataset and creating train-test split": [[0, "Loading-the-UCI-breast-cancer-dataset-and-creating-train-test-split"]], "Training the models": [[0, "Training-the-models"], [1, "Training-the-models"]], "Statistical testing": [[0, "Statistical-testing"], [1, "Statistical-testing"]], "Own metrics": [[0, "Own-metrics"]], "Comparing two regression models using stambo": [[1, "Comparing-two-regression-models-using-stambo"]], "Loading the diabetes dataset and creating train-test split": [[1, "Loading-the-diabetes-dataset-and-creating-train-test-split"]], "Comparing two samples using stambo": [[2, "Comparing-two-samples-using-stambo"]], "Importing the libraries": [[2, "Importing-the-libraries"]], "Data generation": [[2, "Data-generation"]], "Sample comparison": [[2, "Sample-comparison"]], "LaTeX report": [[2, "LaTeX-report"]], "Welcome to stambo\u2019s documentation!": [[3, "welcome-to-stambo-s-documentation"]], "Documentation:": [[3, null]], "Examples:": [[3, null]], "Metrics": [[4, "module-stambo.metrics"]], "Main functionality": [[5, "main-functionality"]]}, "indexentries": {"ap (class in stambo.metrics)": [[4, "stambo.metrics.AP"]], "bacc (class in stambo.metrics)": [[4, "stambo.metrics.BACC"]], "f1score (class in stambo.metrics)": [[4, "stambo.metrics.F1Score"]], "mae (class in stambo.metrics)": [[4, "stambo.metrics.MAE"]], "mcc (class in stambo.metrics)": [[4, "stambo.metrics.MCC"]], "mse (class in stambo.metrics)": [[4, "stambo.metrics.MSE"]], "metric (class in stambo.metrics)": [[4, "stambo.metrics.Metric"]], "qkappa (class in stambo.metrics)": [[4, "stambo.metrics.QKappa"]], "rocauc (class in stambo.metrics)": [[4, "stambo.metrics.ROCAUC"]], "module": [[4, "module-stambo.metrics"], [5, "module-stambo"]], "stambo.metrics": [[4, "module-stambo.metrics"]], "compare_models() (in module stambo)": [[5, "stambo.compare_models"]], "stambo": [[5, "module-stambo"]], "to_latex() (in module stambo)": [[5, "stambo.to_latex"]], "two_sample_test() (in module stambo)": [[5, "stambo.two_sample_test"]]}})