0804a380 <test_allocation_failure>:
 804a380:	55                   	push   %ebp
 804a381:	89 e5                	mov    %esp,%ebp
 804a383:	83 ec 40             	sub    $0x40,%esp
 804a386:	8b 45 08             	mov    0x8(%ebp),%eax
 804a389:	89 45 fc             	mov    %eax,-0x4(%ebp)
 804a38c:	c7 45 e8 00 00 00 00 	movl   $0x0,-0x18(%ebp)
 804a393:	c7 45 f8 ff ff ff ff 	movl   $0xffffffff,-0x8(%ebp)
 804a39a:	8b 45 f8             	mov    -0x8(%ebp),%eax
 804a39d:	89 45 f4             	mov    %eax,-0xc(%ebp)
 804a3a0:	8b 45 f4             	mov    -0xc(%ebp),%eax
 804a3a3:	c1 e8 04             	shr    $0x4,%eax
 804a3a6:	05 01 00 00 00       	add    $0x1,%eax
 804a3ab:	89 45 f0             	mov    %eax,-0x10(%ebp)
 804a3ae:	e8 cd e6 ff ff       	call   8048a80 <__errno@plt>
 804a3b3:	b9 10 00 00 00       	mov    $0x10,%ecx
 804a3b8:	c7 00 00 00 00 00    	movl   $0x0,(%eax)
 804a3be:	8b 45 f0             	mov    -0x10(%ebp),%eax
 804a3c1:	89 04 24             	mov    %eax,(%esp)
 804a3c4:	c7 44 24 04 10 00 00 	movl   $0x10,0x4(%esp)
 804a3cb:	00 
 804a3cc:	89 4d e4             	mov    %ecx,-0x1c(%ebp)
 804a3cf:	e8 bc e8 ff ff       	call   8048c90 <calloc@plt>
 804a3d4:	89 45 ec             	mov    %eax,-0x14(%ebp)
 804a3d7:	81 7d ec 00 00 00 00 	cmpl   $0x0,-0x14(%ebp)
 804a3de:	0f 84 37 00 00 00    	je     804a41b <test_allocation_failure+0x9b>
 804a3e4:	8d 05 58 e5 04 08    	lea    0x804e558,%eax
 804a3ea:	05 b0 00 00 00       	add    $0xb0,%eax
 804a3ef:	8d 0d 10 c8 04 08    	lea    0x804c810,%ecx
 804a3f5:	89 04 24             	mov    %eax,(%esp)
 804a3f8:	89 4c 24 04          	mov    %ecx,0x4(%esp)
 804a3fc:	e8 3f e8 ff ff       	call   8048c40 <fprintf@plt>
 804a401:	8b 4d ec             	mov    -0x14(%ebp),%ecx
 804a404:	89 0c 24             	mov    %ecx,(%esp)
 804a407:	89 45 e0             	mov    %eax,-0x20(%ebp)
 804a40a:	e8 d1 e8 ff ff       	call   8048ce0 <free@plt>
 804a40f:	c7 45 e8 01 00 00 00 	movl   $0x1,-0x18(%ebp)
 804a416:	e9 35 00 00 00       	jmp    804a450 <test_allocation_failure+0xd0>
 804a41b:	e8 60 e6 ff ff       	call   8048a80 <__errno@plt>
 804a420:	8b 00                	mov    (%eax),%eax
 804a422:	89 45 dc             	mov    %eax,-0x24(%ebp)
 804a425:	e8 56 e6 ff ff       	call   8048a80 <__errno@plt>
 804a42a:	8b 00                	mov    (%eax),%eax
 804a42c:	89 04 24             	mov    %eax,(%esp)
 804a42f:	e8 ec e8 ff ff       	call   8048d20 <strerror@plt>
 804a434:	8d 0d 54 c8 04 08    	lea    0x804c854,%ecx
 804a43a:	89 0c 24             	mov    %ecx,(%esp)
 804a43d:	8b 4d dc             	mov    -0x24(%ebp),%ecx
 804a440:	89 4c 24 04          	mov    %ecx,0x4(%esp)
 804a444:	89 44 24 08          	mov    %eax,0x8(%esp)
 804a448:	e8 93 e7 ff ff       	call   8048be0 <printf@plt>
 804a44d:	89 45 d8             	mov    %eax,-0x28(%ebp)
 804a450:	e8 2b e6 ff ff       	call   8048a80 <__errno@plt>
 804a455:	c7 00 00 00 00 00    	movl   $0x0,(%eax)
 804a45b:	8b 45 f4             	mov    -0xc(%ebp),%eax
 804a45e:	2d ff 0f 00 00       	sub    $0xfff,%eax
 804a463:	89 04 24             	mov    %eax,(%esp)
 804a466:	e8 c5 e6 ff ff       	call   8048b30 <malloc@plt>
 804a46b:	89 45 ec             	mov    %eax,-0x14(%ebp)
 804a46e:	81 7d ec 00 00 00 00 	cmpl   $0x0,-0x14(%ebp)
 804a475:	0f 84 21 00 00 00    	je     804a49c <test_allocation_failure+0x11c>
 804a47b:	8d 05 85 c8 04 08    	lea    0x804c885,%eax
 804a481:	89 04 24             	mov    %eax,(%esp)
 804a484:	e8 57 e7 ff ff       	call   8048be0 <printf@plt>
 804a489:	8b 4d ec             	mov    -0x14(%ebp),%ecx
 804a48c:	89 0c 24             	mov    %ecx,(%esp)
 804a48f:	89 45 d4             	mov    %eax,-0x2c(%ebp)
 804a492:	e8 49 e8 ff ff       	call   8048ce0 <free@plt>
 804a497:	e9 35 00 00 00       	jmp    804a4d1 <test_allocation_failure+0x151>
 804a49c:	e8 df e5 ff ff       	call   8048a80 <__errno@plt>
 804a4a1:	8b 00                	mov    (%eax),%eax
 804a4a3:	89 45 d0             	mov    %eax,-0x30(%ebp)
 804a4a6:	e8 d5 e5 ff ff       	call   8048a80 <__errno@plt>
 804a4ab:	8b 00                	mov    (%eax),%eax
 804a4ad:	89 04 24             	mov    %eax,(%esp)
 804a4b0:	e8 6b e8 ff ff       	call   8048d20 <strerror@plt>
 804a4b5:	8d 0d d0 c8 04 08    	lea    0x804c8d0,%ecx
 804a4bb:	89 0c 24             	mov    %ecx,(%esp)
 804a4be:	8b 4d d0             	mov    -0x30(%ebp),%ecx
 804a4c1:	89 4c 24 04          	mov    %ecx,0x4(%esp)
 804a4c5:	89 44 24 08          	mov    %eax,0x8(%esp)
 804a4c9:	e8 12 e7 ff ff       	call   8048be0 <printf@plt>
 804a4ce:	89 45 cc             	mov    %eax,-0x34(%ebp)
 804a4d1:	8b 45 e8             	mov    -0x18(%ebp),%eax
 804a4d4:	85 c0                	test   %eax,%eax
 804a4d6:	0f 95 c1             	setne  %cl
 804a4d9:	0f b6 c1             	movzbl %cl,%eax
 804a4dc:	83 c4 40             	add    $0x40,%esp
 804a4df:	5d                   	pop    %ebp
 804a4e0:	c3                   	ret    
 804a4e1:	90                   	nop
 804a4e2:	90                   	nop
 804a4e3:	90                   	nop
 804a4e4:	90                   	nop
 804a4e5:	90                   	nop
 804a4e6:	90                   	nop
 804a4e7:	90                   	nop
 804a4e8:	90                   	nop
 804a4e9:	90                   	nop
 804a4ea:	90                   	nop
 804a4eb:	90                   	nop
 804a4ec:	90                   	nop
 804a4ed:	90                   	nop
 804a4ee:	90                   	nop
 804a4ef:	90                   	nop

