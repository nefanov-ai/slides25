#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#define MAX_INPUT_LENGTH 256
#define MAX_ARGS 10

void trim_whitespace(char *str) {
    int i = strlen(str) - 1;
    while (i >= 0 && (str[i] == ' ' || str[i] == '\n' || str[i] == '\t')) {
        str[i] = '\0';
        i--;
    }
}

void parse_command(char *input, char **command, char **args, int *arg_count) {
    char *paren = strchr(input, '(');
    if (paren == NULL) {
        *command = strdup(input);
        *arg_count = 0;
        return;
    }

    *command = strndup(input, paren - input);
    
    char *args_start = paren + 1;
    char *args_end = strchr(input, ')');
    if (args_end == NULL) {
        *arg_count = 0;
        return;
    }

    char args_str[MAX_INPUT_LENGTH];
    strncpy(args_str, args_start, args_end - args_start);
    args_str[args_end - args_start] = '\0';

    *arg_count = 0;
    char *token = strtok(args_str, ",");
    while (token != NULL && *arg_count < MAX_ARGS) {
        trim_whitespace(token);
        args[*arg_count] = strdup(token);
        (*arg_count)++;
        token = strtok(NULL, ",");
    }
}

int execute_syscall(char *command, char **args, int arg_count) {
    if (strcmp(command, "fork") == 0) {
        if (arg_count != 0) {
            fprintf(stderr, "fork() takes no arguments\n");
            return -1;
        }
        pid_t pid = fork();
        printf("fork() returned %d\n", pid);
        if (pid > 0) {
            wait(NULL); // Wait for child to prevent zombies
        }
        return pid;
    }
    else if (strcmp(command, "setsid") == 0) {
        if (arg_count != 0) {
            fprintf(stderr, "setsid() takes no arguments\n");
            return -1;
        }
        pid_t sid = setsid();
        printf("setsid() returned %d\n", sid);
        return sid;
    }
    else if (strcmp(command, "setpgid") == 0) {
        if (arg_count != 2) {
            fprintf(stderr, "setpgid() requires exactly 2 arguments\n");
            return -1;
        }
        pid_t pid = atoi(args[0]);
        pid_t pgid = atoi(args[1]);
        int ret = setpgid(pid, pgid);
        printf("setpgid(%d, %d) returned %d\n", pid, pgid, ret);
        return ret;
    }
    else if (strcmp(command, "getpid") == 0) {
        if (arg_count != 0) {
            fprintf(stderr, "getpid() takes no arguments\n");
            return -1;
        }
        pid_t pid = getpid();
        printf("getpid() returned %d\n", pid);
        return pid;
    }
    else if (strcmp(command, "getppid") == 0) {
        if (arg_count != 0) {
            fprintf(stderr, "getppid() takes no arguments\n");
            return -1;
        }
        pid_t ppid = getppid();
        printf("getppid() returned %d\n", ppid);
        return ppid;
    }
    else if (strcmp(command, "getsid") == 0) {
        if (arg_count > 1) {
            fprintf(stderr, "getsid() takes at most 1 argument\n");
            return -1;
        }
        pid_t pid = 0;
        if (arg_count == 1) {
            pid = atoi(args[0]);
        }
        pid_t sid = getsid(pid);
        printf("getsid(%d) returned %d\n", pid, sid);
        return sid;
    }
    else if (strcmp(command, "getpgid") == 0) {
        if (arg_count > 1) {
            fprintf(stderr, "getpgid() takes at most 1 argument\n");
            return -1;
        }
        pid_t pid = 0;
        if (arg_count == 1) {
            pid = atoi(args[0]);
        }
        pid_t pgid = getpgid(pid);
        printf("getpgid(%d) returned %d\n", pid, pgid);
        return pgid;
    }
    else if (strcmp(command, "exit") == 0) {
        if (arg_count > 1) {
            fprintf(stderr, "exit() takes at most 1 argument\n");
            return -1;
        }
        int status = 0;
        if (arg_count == 1) {
            status = atoi(args[0]);
        }
        printf("exit(%d)\n", status);
        exit(status);
    }
    else {
        fprintf(stderr, "Unknown command: %s\n", command);
        return -1;
    }
}

int main() {
    char input[MAX_INPUT_LENGTH];
    
    printf("System Call Executor\n");
    printf("Available commands: fork(), setsid(), setpgid(arg1, arg2), getpid(), getppid(), getsid([pid]), getpgid([pid]), exit([status])\n");
    
    while (1) {
        printf("> ");
        if (fgets(input, MAX_INPUT_LENGTH, stdin) == NULL) {
            break; // EOF or error
        }
        
        trim_whitespace(input);
        if (strlen(input) == 0) {
            continue;
        }
        
        char *command = NULL;
        char *args[MAX_ARGS];
        int arg_count = 0;
        
        parse_command(input, &command, args, &arg_count);
        
        if (command != NULL) {
            execute_syscall(command, args, arg_count);
            
            free(command);
            for (int i = 0; i < arg_count; i++) {
                free(args[i]);
            }
        }
    }
    
    return 0;
}
