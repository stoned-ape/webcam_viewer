#pragma comment
#include "common.h"

#include <libusb-1.0/libusb.h>

#define READ_EP 129
#define WRITE_EP 1

void check_libusb(int err, int line, const char *file, const char *str);

#define LIBUSB(call) check_libusb(call,__LINE__,__FILE__,#call)

typedef enum {
	WINT = 0,
	WFLOAT = 1,
	RINT = 2,
	RFLOAT = 3,
	CRC_ERROR = 4,
	WSTR = 5,
	RSTR = 6,
	WREINIT = 8,
	WINTEMP = 9,
	WEXTEMP = 10,
	CRC_ERROR_W = 16,
	WCLEAR_EEPROM = 13,
}rw_type;


typedef struct __attribute__((packed)) {
	uint8_t header;
	uint8_t sc;
	uint8_t rw;
	union {
		uint8_t data[4];
		uint32_t idata;
		float fdata;
	};
	uint8_t crc;
	uint8_t end;
	char padding[32 - 9];
}frame;

static_assert(sizeof(frame) == 32, "frame struct must be 32 bytes");

int write(libusb_device_handle *dh, void *data, int n);
int read(libusb_device_handle *dh, void *data, int n);
void bin_print(uint64_t x);
uint64_t popcount(uint64_t x);
void _endian(void *v, int size);
uint8_t i_crc8(uint64_t x);
uint8_t a_crc8_array(uint8_t *ptr, int num_bytes);
uint8_t a_crc8(uint64_t x);
void test_crc8();
frame make_frame(char cmd, uint8_t rw, uint32_t data);
void print_frame(frame f);
void print_frame32(frame f);
frame send_frame(libusb_device_handle* dh, frame f1);

typedef struct {
	float x, y, z;
}vec3;

vec3 get_accel(libusb_device_handle* dh);
uint32_t get_dist(libusb_device_handle *dh);
float get_volts(libusb_device_handle *dh, bool five_volts);
float get_led_temp(libusb_device_handle *dh,bool board1);

typedef struct {
	bool led1_ok, led2_ok, accel_ok, dist_ok;
}status;

status get_status(libusb_device_handle *dh);
void set_camera_power(libusb_device_handle *dh, bool on);
void set_color_camera_power(libusb_device_handle *dh, bool on);

typedef union {
	struct {
		uint32_t sensor_id : 1, dist : 9, minute : 5, hour : 5, year : 3, day : 5, month : 4;
	};
	int32_t i;
}date;

static_assert(sizeof(date) == 4, "");

date get_calibration_date(libusb_device_handle *dh, bool sensor1);
void set_led_power(libusb_device_handle *dh, bool on);
void set_ff_led_power(libusb_device_handle *dh, bool on);
void set_ff_green_led_state(libusb_device_handle *dh, bool on);
void set_ff_magenta_led_state(libusb_device_handle *dh, bool on);

enum led_color_cmd :char {
	CC_BLUE = 'F',
	CC_GREEN = 'G',
	CC_AMBER = 'H',
	CC_RED = 'I',
	CC_DEEP_RED = 'I',
	CC_FAR_RED = 'J',
	CC_NIR1 = 'K',
	CC_NIR2 = 'L',
	_CC_NONE = 0,
};

enum led_color :char {
	C_BLUE = 0,
	C_GREEN = 1,
	C_AMBER = 2,
	C_DEEP_RED = 3,
	C_FAR_RED = 4,
	C_NIR1 = 5,
	C_NIR2 = 6,
	_C_NONE = 7,
	C_RED = 8,
};

enum led_color_cmd color2cmd(enum led_color c);
void set_led_state(libusb_device_handle *dh, enum led_color c, bool on);
void set_led_current(libusb_device_handle *dh, enum led_color c, uint8_t x);
void reinit_chips(libusb_device_handle *dh);
void raw_led_test(libusb_device_handle *dh);
bool poll_button(libusb_device_handle *dh);
void get_firmware_version(libusb_device_handle *dh, char *out, int len);
void set_all_led_currents(libusb_device_handle *dh);
frame set_led_groups(libusb_device_handle *dh,bool white,bool visible,bool nir);
frame get_led_groups(libusb_device_handle *dh);
frame set_all_led_currents(libusb_device_handle *dh,int v);
frame get_all_led_currents(libusb_device_handle *dh);
_float2x3 read_gyroscope(libusb_device_handle *dh);
void stm_disconnect(libusb_device_handle *dh);
libusb_device_handle *stm_connect();