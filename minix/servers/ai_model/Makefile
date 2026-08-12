# Makefile for ai_model inference service.
.include <bsd.own.mk>

PROG=	ai_model
SRCS=	ai_model.c ai_inference.c

DPADD+=	${LIBSYS} ${LIBM} ${LIBC}
LDADD+=	-lsys -lm -lc

CPPFLAGS+=	-I${NETBSDSRCDIR}/minix/include

.include <minix.service.mk>
