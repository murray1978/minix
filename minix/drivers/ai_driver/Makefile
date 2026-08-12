# Makefile for the ai_driver character driver.
PROG=	ai_driver
SRCS=	ai_driver.c
CFLAGS+=-g
CFLAGS+=-O0

DPADD+=	${LIBCHARDRIVER} ${LIBSYS}
LDADD+=	-lchardriver -lsys

.include <minix.service.mk>
