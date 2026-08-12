080491d0 <test_allocation_failure>:
 80491d0:	55                   	push   %ebp
 80491d1:	89 e5                	mov    %esp,%ebp
 80491d3:	83 ec 14             	sub    $0x14,%esp
 80491d6:	c7 45 fc ff ff ff ff 	movl   $0xffffffff,-0x4(%ebp)
 80491dd:	8b 45 fc             	mov    -0x4(%ebp),%eax
 80491e0:	e8 3b de 01 00       	call   8067020 <__errno>
 80491e5:	c7 00 00 00 00 00    	movl   $0x0,(%eax)
 80491eb:	c7 44 24 0c b8 a8 07 	movl   $0x807a8b8,0xc(%esp)
 80491f2:	08 
 80491f3:	c7 44 24 08 01 00 00 	movl   $0x1,0x8(%esp)
 80491fa:	00 
 80491fb:	c7 44 24 04 43 00 00 	movl   $0x43,0x4(%esp)
 8049202:	00 
 8049203:	c7 04 24 79 0b 07 08 	movl   $0x8070b79,(%esp)
 804920a:	e8 91 b7 01 00       	call   80649a0 <fwrite>
 804920f:	e8 0c de 01 00       	call   8067020 <__errno>
 8049214:	c7 00 00 00 00 00    	movl   $0x0,(%eax)
 804921a:	c7 04 24 d0 11 07 08 	movl   $0x80711d0,(%esp)
 8049221:	e8 4a 44 00 00       	call   804d670 <puts>
 8049226:	b8 01 00 00 00       	mov    $0x1,%eax
 804922b:	83 c4 14             	add    $0x14,%esp
 804922e:	5d                   	pop    %ebp
 804922f:	c3                   	ret    

