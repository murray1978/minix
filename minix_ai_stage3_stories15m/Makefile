PROG=minix_llama
SRCS=minix_llama.c
MAN=

WARNS?=4
CFLAGS+=-O2
CFLAGS+=-std=c99
CFLAGS+=-D_POSIX_C_SOURCE=200112L
CFLAGS+=-D_LARGEFILE_SOURCE
LDADD+=-lm
DPADD+=${LIBM}

.PHONY: testDriver
testDriver: test_driver.c
	${CC} ${CFLAGS} -o test_driver test_driver.c

.include <bsd.prog.mk>
