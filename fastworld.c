/* fastworld.c -- compiled hot-path for span scanning.
 *
 * Provides fast_find_spans(text, pool_csv) -> str
 *   text      : the narrative
 *   pool_csv  : '|'-separated words to locate (e.g. "kitchen|bedroom|...")
 *   returns   : "word<TAB>start,end;start,end\n..." for each matched word
 *
 * Word boundaries match the Python reference (_py_detect_spans): a match is
 * only whole-word (alnum char before/after disqualifies). This is the same
 * contract detect_spans() expects; swap BACKEND="c" in world_state.py to use
 * it. It is compiled to fastworld.so and loaded via ctypes.
 */
#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

static int iswc(unsigned char c) { return isalnum(c); }

static PyObject *fast_find_spans(PyObject *self, PyObject *args) {
    const char *text;
    const char *poolcsv;
    if (!PyArg_ParseTuple(args, "ss", &text, &poolcsv))
        return NULL;

    size_t tlen = strlen(text);
    char *out = malloc(1);
    size_t cap = 1, len = 0;
    if (out) out[0] = '\0';

    const char *p = poolcsv;
    char wbuf[128];
    while (*p) {
        const char *q = p;
        while (*q && *q != '|') q++;
        size_t wlen = (size_t)(q - p);
        if (wlen > 0 && wlen < sizeof(wbuf) && wlen <= tlen) {
            memcpy(wbuf, p, wlen);
            wbuf[wlen] = '\0';
            const char *pos = text;
            while ((pos = strstr(pos, wbuf)) != NULL) {
                int ok = 1;
                if (pos > text && iswc((unsigned char)pos[-1])) ok = 0;
                const char *after = pos + wlen;
                if (*after && iswc((unsigned char)*after)) ok = 0;
                if (ok) {
                    int start = (int)(pos - text);
                    int end = start + (int)wlen;
                    char buf[512];
                    int n = snprintf(buf, sizeof(buf), "%.*s\t%d,%d;\n",
                                     (int)wlen, wbuf, start, end);
                    if (len + (size_t)n + 1 > cap) {
                        cap = cap * 2 + (size_t)n + 1;
                        char *tmp = realloc(out, cap);
                        if (!tmp) { free(out); return PyUnicode_FromString(""); }
                        out = tmp;
                    }
                    memcpy(out + len, buf, (size_t)n);
                    len += (size_t)n;
                    out[len] = '\0';
                }
                pos += wlen;
            }
        }
        p = (*q) ? q + 1 : q;
    }
    PyObject *res = PyUnicode_FromStringAndSize(out ? out : "", len);
    free(out);
    return res;
}

static PyMethodDef methods[] = {
    {"fast_find_spans", fast_find_spans, METH_VARARGS, "fast whole-word span scan"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "fastworld", "compiled span scanner", -1, methods
};

PyMODINIT_FUNC PyInit_fastworld(void) {
    return PyModule_Create(&moduledef);
}
