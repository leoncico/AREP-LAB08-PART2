import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

web_paths = [
    "https://sandcresearch.medium.com/what-is-the-principle-of-neuromechanical-matching-6e214c299dab",
    "https://sandcresearch.medium.com/does-muscle-loss-happen-within-a-training-week-880a986350c4",
    "https://sandcresearch.medium.com/why-slowing-down-the-eccentric-phase-does-not-cause-more-muscle-growth-9d4e6cb7dd83",
    "https://sandcresearch.medium.com/what-is-the-minimum-number-of-stimulating-reps-in-a-workout-that-will-cause-hypertrophy-8ddb487e4cdb",
    "https://sandcresearch.medium.com/how-does-mechanical-tension-cause-muscle-growth-when-bar-speed-is-maximal-on-every-rep-in-the-set-759b8298e377",
    "https://sandcresearch.medium.com/how-can-we-select-exercises-to-fit-different-training-frequencies-4211c847a5e1",
    "https://sandcresearch.medium.com/what-are-the-different-types-of-fatigue-a631442c973d",
    "https://sandcresearch.medium.com/how-should-we-train-the-triceps-c45f78a5e90",
    "https://sandcresearch.medium.com/how-should-we-train-the-quadriceps-31ad002d0ae4",
    "https://sandcresearch.medium.com/how-can-we-best-train-the-hamstrings-1307fc6be59c",
    "https://sandcresearch.medium.com/how-should-we-train-the-gluteus-maximus-ac35d1bd3c39",
    "https://sandcresearch.medium.com/how-should-we-train-the-calf-muscles-e700eb37e8c0",
    "https://sandcresearch.medium.com/how-many-stimulating-reps-are-there-in-each-set-to-failure-9d179f594dd",
    "https://sandcresearch.medium.com/what-is-the-maximum-number-of-stimulating-reps-that-we-can-do-in-a-workout-for-a-muscle-group-9379d91bf2c",
    "https://sandcresearch.medium.com/how-should-we-train-the-deltoids-f00d9c5388e2",
    "https://sandcresearch.medium.com/how-should-we-train-the-trapezius-1075284e05ee",
    "https://sandcresearch.medium.com/how-should-we-train-the-pectoralis-major-3b302dfbda1e"
]

# Cargar y procesar los documentos
loader = WebBaseLoader(
    web_paths=web_paths,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "du", "ag", "dv", "bf", "ak", "am", "an", "ao", "ap", "aq", "ar", "as", "at", "dw", "dt", "ab", "dx", "dy", "dz", "ea", "eb", "ec", "ed", "ee", "ef", "eg", "eh", "ei", "ej", "ek", "el", "em", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ew", "ex", "ey", "ez", "fa", "fb", "fc", "fd", "bm", "fe", "ff", "ax", "af", "ah", "ai", "aj", "al", "ac", "ae", "au", "av", "aw", "ay", "az", "ba", "bb", "bc", "bd", "bn", "bo", "be", "bg", "bh", "bi", "bj", "bk", "bl", "fg", "fh", "fi", "fj", "fk", "fl", "fm", "fn", "fo", "fp", "fq", "by", "bz", "ca", "cx", "fr", "fs", "ft", "ra", "sp", "sq", "sr", "ss", "st", "su", "gm", "sv", "sw", "sx", "sy", "sz", "ta", "tb", "tc", "lc", "td", "te", "tf", "tg", "th", "ti", "tj", "tk", "tl", "fu", "fv", "fw", "fx", "fy", "cb", "ci", "fz", "ga", "gb", "gc", "gi", "gj", "gk", "gl", "mh", "mi", "to", "speechify-ignore", "qh", "lo", "act", "qo", "tt", "tu", "gn", "go", "gp", "gq", "gr", "pw-post-title", "gs", "gt", "gu", "gv", "gw", "gx", "gy", "gz", "ha", "hb", "hc", "hd", "he", "hf", "hg", "hh", "hi", "hj", "hk", "hl", "hm", "hn", "ho", "hp", "hq", "hr", "hs", "ht", "hu", "tp", "tq", "cp", "hv", "hw", "hx", "hy", "hz", "ia", "ib", "ic", "id", "ie", "dd", "de", "if", "ig", "ih", "ii", "ij", "ik", "il", "im", "in", "io", "ip", "iq", "ir", "is", "cn", "it", "iu", "iv", "iw", "ix", "iy", "iz", "ja", "jb", "jc", "jd", "je", "jf", "jg", "jh", "ji", "jj", "jk", "jl", "jm", "jn", "kd", "ke", "kf", "pw-multi-vote-icon", "kg", "kh", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kp", "kq", "kr", "pw-multi-vote-count", "ks", "kt", "ku", "kv", "kw", "kx", "ky", "tr", "lh", "rf", "rg", "ld", "le", "lf", "lb", "pw-responses-count", "la", "jo", "jp", "jq", "jr", "js", "jt", "ju", "jv", "jw", "jx", "jy", "jz", "ka", "kb", "kc", "lg", "li", "lj", "lk", "ll", "lm", "ln", "lp", "lq", "lr", "ls", "lt", "lu", "lv", "lw", "lx", "ly", "lz", "ma", "mb", "mc", "md", "me", "mf", "mg", "mk", "ml", "mm", "mn", "mo", "mp", "paragraph-image", "mq", "mr", "ms", "mt", "mj", "mu", "pw-post-body-paragraph", "mv", "mw", "mx", "my", "mz", "na", "nb", "nc", "nd", "ne", "nf", "ng", "nh", "ni", "nj", "nk", "nl", "nm", "nn", "no", "np", "nq", "nr", "ns", "nt", "nu", "nv", "nw", "nx", "ny", "nz", "oa", "ob", "oc", "od", "oe", "of", "og", "oh", "oi", "oj", "ok", "ol", "om", "on", "oo", "op", "oq", "or", "os", "ot", "ou", "ov", "ow", "ox", "oy", "oz", "pa", "pb", "pc", "pd", "pe", "pf", "pg", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pp", "pq", "pr", "ps", "pt", "pu", "pv", "pw", "px", "py", "pz", "qa", "qb", "qc", "qd", "qe", "qf", "abl", "acu", "ade", "acw", "acx", "acy", "acz", "qg", "qi", "qj", "qk", "ql", "ge", "qm", "qn", "qp", "qq", "qr", "qs", "qt", "qu", "qv", "qw", "qx", "qy", "qz", "rb", "rc", "rd", "re", "bq", "rh", "ri", "rj", "rk", "rl", "bx", "cl", "rm", "ada", "adb", "rp", "adc", "add", "tn", "rt", "ru", "pw-author-name", "se", "sf", "sg", "sh", "pw-follower-count", "si", "sj", "sk", "tv", "tw", "tx", "ty", "tz", "ua", "ub", "uc", "ud", "ue", "uf", "ug", "uh", "ui", "uj", "uk", "ul", "um", "un", "uo", "up", "uq", "ur", "us", "ut", "uu", "uv", "uw", "ux", "uy", "uz", "va", "vb", "vc", "vd", "ve", "vf", "vg", "vh", "vi", "vj", "vk", "vl", "vm", "vn", "vo", "vp", "vq", "vr", "vs", "vt", "vu", "vv", "vw", "vx", "vy", "vz", "wa", "wc", "wd", "we", "wf", "wg", "wh", "wi", "wj", "wk", "wb", "co", "wl", "wm", "wn", "wo", "wp", "wq", "wr", "ws", "wt", "wu", "wv", "ww", "wx", "wy", "wz", "xa", "xb", "xc", "xd", "xe", "xf", "xg", "xh", "xi", "xj", "xk", "xl", "xm", "xn", "xo", "xp", "xq", "xr", "xs", "xt", "xu", "xv", "xw", "xx", "xy", "xz", "ya", "yb", "yc", "yd", "ye", "yh", "yf", "dh", "yg", "dj", "yi", "yj", "yk", "yl", "dk", "dl", "ym", "yn", "yo", "yp", "yq", "yr", "ys", "yt", "yu", "yv", "yw", "yx", "yy", "yz", "za", "zb", "zc", "zd", "ze", "zf", "zg", "zh", "zi", "zj", "zk", "zl", "zm", "zn", "zo", "zp", "zq", "zw", "zx", "zs", "zt", "zu", "zv", "zr", "zy", "zz", "aba", "abb", "abc", "abd", "abe", "abf", "abg", "abh", "abi", "sl", "sm", "sn", "so")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_rag_response(question: str) -> str:
    return rag_chain.invoke(question)