#!/bin/sh

PASS=0
FAIL=0
EXPECT_IPC_DENIED=${AI_DRIVER_EXPECT_IPC_DENIED:-no}

pass() {
	PASS=`expr "$PASS" + 1`
	echo "PASS: $1"
}

fail() {
	FAIL=`expr "$FAIL" + 1`
	echo "FAIL: $1"
}

contains() {
	echo "$1" | grep "$2" >/dev/null 2>&1
}

if [ ! -e /dev/ai_driver ]; then
	echo "FAIL: /dev/ai_driver does not exist"
	exit 1
fi

if [ ! -c /dev/ai_driver ]; then
	echo "FAIL: /dev/ai_driver is not a character device"
	exit 1
fi
pass "device node exists and is character device"

if echo "-v on" > /dev/ai_driver 2>/dev/null; then
	resp=`cat /dev/ai_driver`
	if contains "$resp" "verbose=on"; then
		pass "-v on command"
	else
		fail "-v on response check"
	fi
else
	fail "-v on write failed"
fi

if echo "status" > /dev/ai_driver 2>/dev/null; then
	resp=`cat /dev/ai_driver`
	if contains "$resp" "verbose=on"; then
		pass "status reports verbose=on"
	else
		fail "status missing verbose=on"
	fi
else
	fail "status write failed"
fi

if [ "$EXPECT_IPC_DENIED" = "yes" ]; then
	if echo "i robots need sleep" > /dev/ai_driver 2>/dev/null; then
		fail "denied-permission inference unexpectedly succeeded"
	else
		pass "denied-permission inference rejected"
	fi

	if echo "status" > /dev/ai_driver 2>/dev/null; then
		resp=`cat /dev/ai_driver`
		if contains "$resp" "state=error" && \
		   contains "$resp" "last_error="; then
			pass "driver survived denied IPC and reports error state"
		else
			fail "driver status missing denied-IPC error state"
		fi
	else
		fail "status after denied IPC failed"
	fi
else
	if echo "i robots need sleep" > /dev/ai_driver 2>/dev/null; then
		resp=`cat /dev/ai_driver`
		if contains "$resp" "backend=ai_model" && \
		   contains "$resp" "response="; then
			pass "plain prompt inference"
		else
			fail "plain prompt response content"
		fi
	else
		fail "plain prompt write failed"
	fi

	if echo 'i "robots need sleep"' > /dev/ai_driver 2>/dev/null; then
		resp=`cat /dev/ai_driver`
		if contains "$resp" "backend=ai_model" && \
		   contains "$resp" "response="; then
			pass "quoted prompt inference"
		else
			fail "quoted prompt response content"
		fi
	else
		fail "quoted prompt write failed"
	fi
fi

if echo "reset" > /dev/ai_driver 2>/dev/null; then
	resp=`cat /dev/ai_driver`
	if contains "$resp" "reset=ok"; then
		pass "reset command"
	else
		fail "reset response check"
	fi
else
	fail "reset write failed"
fi

if printf "bogus\n" > /dev/ai_driver 2>/dev/null; then
	fail "invalid command unexpectedly succeeded"
else
	pass "invalid command rejected"
fi

BIGCMD=`awk 'BEGIN { for(i=0;i<1030;i++) printf "x"; printf "\n"; }'`
if printf "%s" "$BIGCMD" > /dev/ai_driver 2>/dev/null; then
	fail "oversized command unexpectedly succeeded"
else
	pass "oversized command rejected"
fi

if [ "$FAIL" -eq 0 ]; then
	echo "SUMMARY: PASS ($PASS checks)"
	if [ "$EXPECT_IPC_DENIED" = "yes" ]; then
		echo "NOTE: restart ai_driver/ai_model with corrected IPC policy and rerun this script without AI_DRIVER_EXPECT_IPC_DENIED=yes to verify recovery success."
	fi
	exit 0
else
	echo "SUMMARY: FAIL ($FAIL failed, $PASS passed)"
	exit 1
fi
