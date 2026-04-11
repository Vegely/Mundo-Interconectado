import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm


RULES = [
    ("Cloud / DevOps", [
        "aws-cdk", "aws_cdk", "boto3", "botocore", "s3fs", "gcsfs", "adlfs",
        "aiobotocore", "google-cloud", "google-auth", "google-api",
        "google-resumable", "azure", "dropbox", "paramiko", "smbprotocol",
        "awscrt", "pykube", "kopf", "sagemaker", "modal", "coiled", "submitit",
        "apache-beam", "google-apitools", "oci", "aiooss", "ocifs",
        "opentelemetry", "prometheus", "grpcio", "proto-plus", "stone",
        "keyrings.google", "awscli", "awslogs", "apache-libcloud",
        "aliyunstoreplugin", "moto", "kubernetes", "docker", "pulumi",
        "cloudformation", "cdk", "celery", "dramatiq", "arq", "apscheduler",
        "msrestazure", "adal", "pyspnego", "slack-sdk", "discord.py",
        "matrix-client", "smpplib", "irc", "apprise", "confluent-kafka",
        "kafka-python", "faust", "aio-pika", "pika", "amqp",
        "grpc-google-iam", "grpc-stubs", "grpcio-tools", "grpcio-status",
        "protoc-gen-openapiv2", "protobuf", "e2b-code-interpreter",
        "oxylabs", "spider-client", "firecrawl",
    ]),
    ("IA / ML", [
        "tensorflow", "keras", "torch", "scikit-learn", "sklearn",
        "transformer", "huggingface", "huggingface_hub", "openai", "langchain",
        "sentence-transform", "ml_dtypes", "ml-dtypes", "optuna", "deepchem",
        "dgl", "dgllife", "botorch", "allennlp", "xgboost", "lightgbm",
        "catboost", "mlflow", "mlrun", "deepeval", "promptflow", "google-genai",
        "datasets", "accelerate", "vllm", "onnxruntime", "fastai", "gymnasium",
        "deepspeed", "timm", "tokenizers", "safetensors", "torchaudio",
        "torchvision", "pytorch_lightning", "torchx", "torchserve",
        "tensorboard", "tensorflow-probability", "tensorflow-addons",
        "tensorflow-transform", "tensorflow-hub", "skl2onnx", "tf2onnx",
        "pyod", "xitorch", "rna-fm", "modlamp", "ogb", "liger-kernel",
        "llm-blender", "neo4j-graphrag", "langchain-memgraph",
        "langchain-neo4j", "rank-bm25", "bayesian-optimization", "ax-platform",
        "scikit-optimize", "nevergrad", "pyswarms", "skorch", "embeddings",
        "evaluate", "nltk", "sacremoses", "sacrebleu", "pyctcdecode",
        "phonemizer", "tiktoken", "blobfile", "codecarbon", "dm_tree", "nixl",
        "datatable", "facets-overview", "optimum-benchmark", "hf-xet", "mcp",
        "opik", "langchain-experimental", "memgraph-toolbox", "pinecone",
        "semchunk", "docling", "math-verify", "latex2sympy", "ptflops",
        "fast-pareto", "fcmaes", "hiplot", "bayes-optim", "image-quality",
        "keras-preprocessing", "linear-tree", "pydoe", "sobol-seq",
        "trustregion", "cudf", "pylibcudf", "rmm-cu12", "nvtx", "cupy",
        "ray", "bodo", "daft", "google-cloud-aiplatform",
        "google-cloud-batch", "fschat", "pycocoevalcap", "pyglove",
        "beaker-py", "mldesigner", "mip", "pyomo", "desdeo", "jmetalpy",
        "pymoo", "pfns", "moocore", "directsearch", "olymp", "anthropic",
        "autogen", "ag2", "ai21", "adapters", "ale-py", "ale_py", "aeon",
        "albumentations", "anndata", "bert-score", "bert_score",
        "arize", "causalml", "cerebras-cloud-sdk",
        "scikit-base", "pytorch_optimizer", "optuna-integration",
        "scikit-umfpack",
    ]),
    ("Testing", [
        "pytest", "hypothesis", "-mock", "_mock", "tox", "behave",
        "coverage", "coveralls", "vcrpy", "responses", "betamax",
        "nox", "testpath", "pyfakefs", "nbmake", "nbval",
        "pytest_timeout", "nbproject-test", "deepeval", "pytest_codspeed",
        "syrupy", "allpairspy", "asv", "pook", "pyperf", "deal", "icontract",
        "allure", "parameterized", "locust", "schemathesis",
    ]),
    ("Documentación", [
        "sphinx", "myst", "nbsphinx", "numpydoc", "pandoc",
        "rst2pdf", "rstcheck", "blurb", "towncrier", "reno",
        "mkdocs", "intersphinx", "furo", "pydata-sphinx",
        "nbconvert", "nbdime", "jupytext", "jupyterlite", "jupyter-book",
        "quartodoc", "griffe", "autodoc", "autodocsumm",
        "readmemaker", "python-docs-theme", "rst2txt", "commonmark",
        "recommonmark", "mdit-py", "linkify", "markdown-it",
        "markdown-include", "mdformat", "pymdown", "pybtex", "shibuya",
        "sphinxcontrib", "sphinxext", "ablog", "sphinx-",
        "mkdocstrings", "mkdocs-gen-files", "mkdocs-literate",
        "mkdocs-section", "mkdocs-jupyter", "mkdocs-material",
        "mkdocs-redirects", "mkdocs-git", "mkdocs-llmstxt",
    ]),
    ("Base de Datos", [
        "sqlalchemy", "pymysql", "redis", "pymongo", "psycopg",
        "aiomysql", "aiosqlite", "oracledb", "databricks-sql",
        "clickhouse", "snowflake", "trino", "singlestore",
        "adbc", "mysql-connector", "pyarango", "cassandra",
        "elasticsearch", "pymilvus", "pycouchdb", "pyexasol",
        "anysqlite", "geoalchemy", "graphdatascience",
        "neo4j", "delta-spark", "apache-flink", "sqlglot",
        "unitycatalog", "sqlalchemy-pytds", "supabase", "postgrest",
        "storage3", "supabase-auth", "alembic", "motor", "beanie",
        "peewee", "mongoengine", "databases", "tortoise-orm",
        "lancedb", "chromadb", "qdrant", "weaviate", "milvus",
        "pinecone", "vespa", "typesense", "influxdb",
    ]),
    ("Ciencia de Datos", [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "jupyter", "notebook", "ipython", "ipykernel", "ipywidgets",
        "jupyterlab", "statsmodels", "xarray", "altair", "bokeh",
        "plotly", "panel", "polars", "duckdb", "fastparquet",
        "narwhals", "bottleneck", "netcdf4", "h5netcdf",
        "zarr", "pooch", "cartopy", "geopandas", "shapely", "geopy",
        "folium", "mapclassify", "geoarrow", "modin", "pyspark",
        "ibis-framework", "cubed", "flox", "sparse", "pydap",
        "xlrd", "pyreadstat", "pyiceberg", "nanoarrow", "fastexcel",
        "deltalake", "fastavro", "marimo", "altair-tiles", "vegafusion",
        "great-tables", "xport", "datapackage", "fecfile",
        "contourpy", "cycler", "fonttools", "pyshp", "fiona",
        "owslib", "ipyleaflet", "plotnine", "pygal", "librosa",
        "dask", "distributed", "partd", "numcodecs", "blosc2", "fsspec",
        "matrepr", "finch-tensor", "cubed-xarray", "numpy_groupies",
        "db-dtypes", "pandas-gbq", "bigquery-magics", "pandas-datareader",
        "numba", "llvmlite", "sympy", "networkx", "mpmath", "gmpy2",
        "hvplot", "holoviews", "datashader", "param", "pyviz-comms",
        "pymc", "arviz", "bambi", "pgmpy", "pomegranate", "hmmlearn",
        "gensim", "top2vec", "bertopic", "tsfresh", "sktime", "tslearn",
        "darts", "prophet", "kats", "astropy", "anndata", "scanpy",
        "rasterio", "rioxarray", "pyproj", "pint", "uncertainties",
        "vispy", "pyqtgraph", "scikit-image", "imageio",
    ]),
    ("Desarrollo Web", [
        "fastapi", "django", "flask", "starlette", "jinja2",
        "aiohttp", "httpx", "requests", "gunicorn", "scrapy",
        "beautifulsoup", "html5lib", "lxml", "httpcore", "httpbin",
        "httplib2", "twisted", "sanic", "gevent", "uvloop", "anyio",
        "trio", "playwright", "firecrawl", "newspaper3k",
        "websocket", "simple-websocket", "pyngrok", "uvicorn",
        "quart", "hypercorn", "selenium", "flask-restx",
        "requests-html", "asgiref", "httpx-aiohttp", "aiostream",
        "aioquic", "email-validator", "webob", "pastedeploy",
        "shiny", "htmltools", "faicons", "streamlit", "gradio",
        "authlib", "pyjwt", "oauth2client", "requests-oauthlib",
        "asgi-csrf", "aiohttp_cors", "aioresponses",
        "requests-cache", "requests-toolbelt", "kombu",
        "django-pgtrigger", "django-schema", "dj-database-url",
        "graphene", "strawberry-graphql", "ariadne",
        "marshmallow", "webargs", "apispec", "werkzeug",
        "itsdangerous", "passlib", "argon2", "social-auth",
        "oauthlib", "pysaml2", "httpretty", "respx",
    ]),
    ("Herramientas Dev", [
        "setuptools", "wheel", "build", "hatch", "flit", "pip-",
        "twine", "importlib", "virtualenv", "meson", "pkginfo",
        "pyinstaller", "pre-commit", "black", "pylint", "mypy",
        "ruff", "flake8", "isort", "pyink", "pyright", "pylance",
        "pyroma", "check-manifest", "check-sdist", "codespell",
        "typeguard", "pdm", "pipenv", "copier", "dunamai",
        "safety", "sentry_sdk", "codecov", "coveralls",
        "setuptools_scm", "jaraco", "zipp", "pathspec", "gitpython",
        "click", "typer", "argcomplete", "rich", "tqdm", "colorlog",
        "humanize", "icecream", "pysnooper", "objgraph", "memray",
        "line_profiler", "pyproject", "hatchling", "pip-run", "pip-tools",
        "unearth", "pbs-installer", "modernize", "ufmt", "usort", "blackdoc",
        "hacking", "parver", "dom-toml", "toml-cli", "dynaconf", "anyconfig",
        "wrapt", "deprecated", "tenacity", "taskipy", "setproctitle",
        "psutil", "dill", "execnet", "mkinit", "xinspect", "scriptconfig",
        "progiter", "munch", "pydash", "blessed", "alive_progress",
        "progressbar2", "rich_argparse", "actionlint", "gcovr",
        "safety-schemas", "dparse", "spdx-tools", "abi3audit", "pypinfo",
        "tuna", "psleak", "import-linter", "pylama", "enum-tools", "beartype",
        "typeshed", "z3-solver", "pygls", "wasmtime", "wmi", "pyreadline3",
        "xattr", "xdev", "subprocess-tee", "pyee", "pyparsing", "inflect",
        "backports", "tempora", "portend", "secretstorage", "keyring",
        "diff_cover", "isoduration", "uri-template", "rfc3987", "jsonschema",
        "homebrew-pypi-poet", "arpeggio", "lark", "crosshair-tool",
        "python-gitlab", "gspread", "mmh3", "zopfli", "unicodedata2",
        "cairosvg", "olefile", "pillow", "deepmerge", "cattrs",
        "pyyaml", "ruamel", "tomlkit", "tomli", "orjson", "ujson",
        "msgpack", "attrs", "pydantic", "marshmallow", "voluptuous",
        "python-dotenv", "decouple", "environs", "omegaconf",
        "loguru", "structlog", "colorama", "termcolor",
        "pendulum", "arrow", "dateutil", "pytz", "tzlocal", "dateparser",
        "packaging", "plumbum", "sh", "shellingham",
        "pygit2", "dulwich", "ghapi", "pygithub",
    ]),
]


def classify(pkg: str) -> str:
    p = pkg.lower().replace("_", "-")
    for category, keywords in RULES:
        if any(kw in p for kw in keywords):
            return category
    return "Otros"


def load_packages(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def make_pie(counts: Counter, total: int, title: str, output_path: str,
             dpi: int = 300, figsize: tuple = (9, 6.5)) -> None:

    PALETTE = {
        "Otros":           "#888780",
        "IA / ML":         "#7F77DD",
        "Cloud / DevOps":  "#378ADD",
        "Herramientas Dev":"#1D9E75",
        "Ciencia de Datos":"#EF9F27",
        "Desarrollo Web":  "#D4537E",
        "Testing":         "#D85A30",
        "Base de Datos":   "#639922",
        "Documentación":   "#5DCAA5",
    }

    labels_sorted = sorted(counts.keys(), key=lambda k: -counts[k])
    sizes  = [counts[k] for k in labels_sorted]
    colors = [PALETTE.get(k, "#aaaaaa") for k in labels_sorted]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=140,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
        pctdistance=0.78,
    )

    wedges[0].set_radius(1.04)

    legend_labels = [
        f"{k}  —  {counts[k]:,}  ({100*counts[k]/total:.1f}%)"
        for k in labels_sorted
    ]
    patches = [
        mpatches.Patch(facecolor=colors[i], edgecolor="white", linewidth=0.5)
        for i in range(len(labels_sorted))
    ]
    legend = ax.legend(
        patches, legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.5,
        frameon=False,
        handlelength=1.2,
        handleheight=1.2,
        labelspacing=0.65,
    )

    ax.text(0, 0, f"{total:,}\npaquetes", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none"))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=18, color="#222222")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", format="png")
    plt.close()
    print(f"Gráfica guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clasifica paquetes PyPI y genera gráfica de pastel en español.")
    parser.add_argument("--input",   default="cycle_0_packages.txt",
                        help="Fichero con un paquete por línea")
    parser.add_argument("--output",  default="pypi_ciclos_categorias.png",
                        help="Ruta de salida del PNG")
    parser.add_argument("--title",   default="Componente gigante — Ciclo 0\nDistribución por área de desarrollo",
                        help="Título de la gráfica")
    parser.add_argument("--dpi",     type=int, default=300)
    parser.add_argument("--figsize", type=float, nargs=2, default=[9, 6.5],
                        metavar=("W", "H"))
    args = parser.parse_args()

    pkgs   = load_packages(args.input)
    counts = Counter(classify(p) for p in pkgs)
    total  = sum(counts.values())

    print(f"\nPaquetes totales: {total}")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<22} {n:>5}  ({100*n/total:.1f}%)")

    make_pie(counts, total, args.title, args.output,
             dpi=args.dpi, figsize=tuple(args.figsize))


if __name__ == "__main__":
    main()