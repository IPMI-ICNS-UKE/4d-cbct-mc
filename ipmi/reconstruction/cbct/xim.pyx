import struct
from collections import namedtuple

cimport cython
cimport numpy as np

import numpy as np


cdef extern from "stdio.h" nogil:
    ctypedef struct FILE
    cdef FILE *stdin
    cdef FILE *stdout
    cdef FILE *stderr

    enum: FOPEN_MAX
    enum: FILENAME_MAX
    FILE *fopen(const char *filename, const char *opentype)
    FILE *freopen(const char *filename, const char *opentype, FILE *stream)
    FILE *fdopen(int fdescriptor, const char *opentype)
    int  fclose(FILE *stream)
    int  remove(const char *filename)
    int  rename(const char *oldname, const char *newname)
    FILE *tmpfile()

    int remove(const char *pathname)
    int rename(const char *oldpath, const char *newpath)

    enum: _IOFBF
    enum: _IOLBF
    enum: _IONBF
    int setvbuf(FILE *stream, char *buf, int mode, size_t size)
    enum: BUFSIZ
    void setbuf(FILE *stream, char *buf)

    size_t fread(void *data, size_t size, size_t count, FILE *stream)
    size_t fwrite(const void *data, size_t size, size_t count, FILE *stream)
    int fflush(FILE *stream)

    enum: EOF
    void clearerr(FILE *stream)
    int feof(FILE *stream)
    int ferror(FILE *stream)

    enum: SEEK_SET
    enum: SEEK_CUR
    enum: SEEK_END
    int fseek(FILE *stream, long int offset, int whence)
    void rewind(FILE *stream)
    long int ftell(FILE *stream)

    ctypedef struct fpos_t:
        pass
    ctypedef const fpos_t const_fpos_t "const fpos_t"
    int fgetpos(FILE *stream, fpos_t *position)
    int fsetpos(FILE *stream, const fpos_t *position)

    int scanf(const char *template, ...)
    int sscanf(const char *s, const char *template, ...)
    int fscanf(FILE *stream, const char *template, ...)

    int printf(const char *template, ...)
    int sprintf(char *s, const char *template, ...)
    int snprintf(char *s, size_t size, const char *template, ...)
    int fprintf(FILE *stream, const char *template, ...)

    void perror(const char *message)

    char *gets(char *s)
    char *fgets(char *s, int count, FILE *stream)
    int getchar()
    int fgetc(FILE *stream)
    int getc(FILE *stream)
    int ungetc(int c, FILE *stream)

    int puts(const char *s)
    int fputs(const char *s, FILE *stream)
    int putchar(int c)
    int fputc(int c, FILE *stream)
    int putc(int c, FILE *stream)

    size_t getline(char ** lineptr, size_t *n, FILE *stream)

@cython.wraparound(False)
@cython.boundscheck(False)
def read_xim(filename):

    headerNames = namedtuple(
        "XimHeader",
        "FileType FileFormatVersion Width Height BitsPerPixel AllocatedBytesPerPixel Compressed"
    )

    headers = {}

    formatstring = "=32sI4sI8s8s16sI16sI16sIIId16sIII4sdddddddddddddddddddddddddddddddddddddddd"
    formatstring = "=8sIIIIII"

    # convert to string (in case of pathlib Path)
    filename = str(filename)
    filename_byte = filename.encode("UTF-8")
    cdef char * fname = filename_byte
    cdef FILE * cfile;
    cfile = fopen(fname, "rb")
    headerLength = 8 + 6 * 4

    cdef char tmp[1024]

    cdef int buffersize
    cdef int footerFields
    cdef fpos_t filePos

    fread(&tmp[0], headerLength, 1, cfile)
    header = headerNames._make(struct.unpack(formatstring, tmp[:headerLength]))

    if header.Compressed:
        fgetpos(cfile, &filePos)
        fread(&buffersize, 4, 1, cfile)
        fseek(cfile, buffersize, 1)
        fread(&buffersize, 4, 1, cfile)
        fseek(cfile, buffersize + 4, 1)
        histogram = read_histogram(cfile)
        footers = read_footers(cfile)
        fsetpos(cfile, &filePos)
        buf = readCompressed4(cfile, header.Width, header.Height)

        fclose(cfile)
        return buf, footers

@cython.wraparound(False)
cdef read_histogram(FILE * cfile):
    cdef int buffersize
    cdef np.ndarray[dtype=np.int32_t, ndim=1] histogram
    fread(&buffersize, 4, 1, cfile)
    histogram = np.zeros((buffersize), dtype=np.int32)
    fread(&histogram[0], 4, buffersize, cfile)

    return histogram

@cython.wraparound(False)
@cython.boundscheck(False)
cdef read_footers(FILE * cfile):
    cdef int footerFields
    cdef int footerType
    cdef np.ndarray[dtype=np.uint8_t, ndim=1] charArr
    cdef int nameLength
    cdef int byteCount
    cdef int intVal
    cdef double dVal
    cdef np.ndarray[dtype=np.int32_t, ndim=1] intArr
    cdef np.ndarray[dtype=np.float64_t, ndim=1] dArr
    fread(&footerFields, 4, 1, cfile)
    footers = {}

    for i in range(footerFields):
        fread(&nameLength, 4, 1, cfile)
        charArr = np.zeros(nameLength, dtype=np.uint8)
        fread(&charArr[0], 1, nameLength, cfile)
        name = "".join(map(chr, charArr))

        fread(&footerType, 4, 1, cfile)
        if footerType == 0:
            fread(&intVal, 4, 1, cfile)
            footers[name] = intVal

        elif footerType == 1:
            fread(&dVal, 8, 1, cfile)
            footers[name] = dVal

        elif footerType == 2:
            fread(&byteCount, 4, 1, cfile)
            charArr = np.zeros(byteCount, dtype=np.uint8)
            if byteCount > 0:
                fread(&charArr[0], 1, byteCount, cfile)
            footers[name] = "".join(map(chr, charArr))

        elif footerType == 3:
            fread(&byteCount, 4, 1, cfile)
            charArr = np.zeros(byteCount, dtype=np.uint8)
            fread(&charArr[0], 1, byteCount, cfile)
            footers[name] = charArr
        elif footerType == 4:
            fread(&byteCount, 4, 1, cfile)
            dArr = np.zeros((byteCount // 8), dtype=np.float64)
            fread(&dArr[0], 8, byteCount // 8, cfile)
        elif footerType == 5:
            fread(&byteCount, 4, 1, cfile)
            intArr = np.zeros((byteCount // 4), dtype=np.int32)
            fread(&intArr[0], 4, byteCount // 4, cfile)
        else:
            print("Unknown footer ftype")
    return footers

@cython.wraparound(False)
@cython.boundscheck(False)
cdef readCompressed4(FILE * cfile, int sizeX, int sizeY):
    cdef np.ndarray[np.uint8_t, ndim=1] pt_lut = np.zeros([sizeX * sizeY],
                                                          dtype=np.uint8)
    cdef int nbytes
    fread(&nbytes, 4, 1, cfile)
    fread(&pt_lut[0], 1, nbytes, cfile)

    cdef np.ndarray[np.uint32_t, ndim=1] buf = np.zeros([sizeX * sizeY],
                                                        dtype=np.uint32)
    fread(&nbytes, 4, 1, cfile)

    # read first row
    fread(&buf[0], 4, sizeX, cfile)

    # read first pixel of second row
    fread(&buf[sizeX], 4, 1, cfile)
    i = sizeX + 1
    cdef unsigned int lut_idx = 0;
    cdef unsigned int lut_off = 0;
    cdef np.uint32_t r11;
    cdef np.uint32_t r12;
    cdef np.uint32_t r21;
    cdef np.uint8_t v;
    cdef np.int8_t dc;
    cdef np.int16_t ds;
    cdef np.uint32_t dl;
    cdef np.int32_t diff = 0;

    cdef int m = 0;
    while i < sizeX * sizeY:
        r11 = buf[i - sizeX - 1]
        r12 = buf[i - sizeX]
        r21 = buf[i - 1]
        v = pt_lut[lut_idx]
        if lut_off == 0:
            v = v & 0x03
            lut_off += 1
        elif lut_off == 1:
            v = (v & 0x0C) >> 2
            lut_off += 1
        elif lut_off == 2:
            v = (v & 0x30) >> 4
            lut_off += 1
        elif lut_off == 3:
            v = (v & 0xC0) >> 6
            lut_off = 0
            lut_idx += 1
        if v == 0:
            fread(&dc, 1, 1, cfile)
            diff = dc
        elif v == 1:
            fread(&ds, 2, 1, cfile)
            diff = ds
        elif v == 2:
            fread(&diff, 4, 1, cfile)
        buf[i] = r21 + r12 + diff - r11
        i += 1
    return np.reshape(buf, [sizeY, sizeX])
