#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AI_DEVICE_PATH "/dev/ai_driver"
#define RESPONSE_BUF_SIZE 8192
#define PROMPT_BUF_SIZE 800
#define COMMAND_BUF_SIZE 1024

static const char *k_cmd_verbose = "-v on\n";
static const char *k_cmd_temperature = "-t 0.9\n";
static const char *k_cmd_top_p = "-p 0.9\n";
static const char *k_cmd_steps = "-n 64\n";
static const char *k_cmd_seed = "-s 1\n";

static int write_command(const char *command)
{
	int fd;
	size_t len;
	ssize_t n;

	fd = open(AI_DEVICE_PATH, O_WRONLY);
	if (fd < 0) {
		perror("open for write");
		return -1;
	}

	len = strlen(command);
	n = write(fd, command, len);
	if (n < 0) {
		perror("write");
		close(fd);
		return -1;
	}
	if ((size_t)n != len) {
		fprintf(stderr, "short write: wrote %ld of %lu bytes\n",
		    (long)n, (unsigned long)len);
		close(fd);
		return -1;
	}

	if (close(fd) != 0) {
		perror("close write fd");
		return -1;
	}

	return 0;
}

static int read_response(char *buffer, size_t size)
{
	int fd;
	size_t used;
	ssize_t n;

	if (size == 0)
		return -1;

	fd = open(AI_DEVICE_PATH, O_RDONLY);
	if (fd < 0) {
		perror("open for read");
		return -1;
	}

	used = 0;
	while (used + 1 < size) {
		n = read(fd, buffer + used, size - used - 1);
		if (n < 0) {
			perror("read");
			close(fd);
			return -1;
		}
		if (n == 0)
			break;
		used += (size_t)n;
	}
	buffer[used] = '\0';

	if (close(fd) != 0) {
		perror("close read fd");
		return -1;
	}

	return 0;
}

static int send_and_print(const char *command)
{
	char response[RESPONSE_BUF_SIZE];

	if (write_command(command) != 0)
		return -1;
	if (read_response(response, sizeof(response)) != 0)
		return -1;

	printf("----- command -----\n%s", command);
	printf("----- response ----\n%s\n", response);
	return 0;
}

int main(void)
{
	char prompt[PROMPT_BUF_SIZE];
	char infer_command[COMMAND_BUF_SIZE];
	size_t len;

	printf("Using %s\n", AI_DEVICE_PATH);
	printf("Fixed model settings:\n");
	printf("  temperature: 0.9\n");
	printf("  top-p: 0.9\n");
	printf("  steps: 64\n");
	printf("  seed: 1\n\n");

	if (send_and_print(k_cmd_verbose) != 0)
		return EXIT_FAILURE;
	if (send_and_print(k_cmd_temperature) != 0)
		return EXIT_FAILURE;
	if (send_and_print(k_cmd_top_p) != 0)
		return EXIT_FAILURE;
	if (send_and_print(k_cmd_steps) != 0)
		return EXIT_FAILURE;
	if (send_and_print(k_cmd_seed) != 0)
		return EXIT_FAILURE;

	printf("Enter prompt text for ai_driver: ");
	fflush(stdout);
	if (fgets(prompt, sizeof(prompt), stdin) == NULL) {
		if (ferror(stdin))
			perror("fgets");
		fprintf(stderr, "no prompt received\n");
		return EXIT_FAILURE;
	}

	len = strlen(prompt);
	while (len > 0 && (prompt[len - 1] == '\n' || prompt[len - 1] == '\r')) {
		prompt[len - 1] = '\0';
		len--;
	}
	if (len == 0) {
		fprintf(stderr, "prompt cannot be empty\n");
		return EXIT_FAILURE;
	}

	if (snprintf(infer_command, sizeof(infer_command), "i %s\n", prompt) >=
	    (int)sizeof(infer_command)) {
		fprintf(stderr, "prompt too long for command buffer\n");
		return EXIT_FAILURE;
	}

	if (send_and_print(infer_command) != 0)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
