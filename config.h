#ifndef CONFIG_H_
#define CONFIG_H_

//#define DEBUG_MODE_

// operating system
#define OS_LINUX
//#define OS_WINDOWS

// some constants
#define MAX_DENSE_TENSOR_DIM 1500
#define MAX_LOG_HASH_LEN 20
// HASH_OMEGA_PERIOD must be of power 2
#define HASH_OMEGA_PERIOD 4

#define MAX_CMD_ARGUMENT_LEN 1000
#define MAX_DOCUMENT_FILES 1000
#define MAX_FILENAME_LENGTH 1000

// for topic model, default parameters
#define DEFAULT_ALPHA 0.1
#define DEFAULT_BETA 0.1

#endif
