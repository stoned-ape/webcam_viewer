#include "usb_api.h"

#include <arpa/inet.h>


void check_libusb(int err, int line, const char *file, const char *str) {
	if (err < 0) {
		fprintf(stderr, "libusb error (%s) on line %d of file %s\n\tLIBUSB(%s)\n",
		// 	libusb_strerror(err),
			"bruh", line, file, str);
		exit(err);
	}
}

int write(libusb_device_handle *dh, void *data, int n) {
	assert(data);
	/*int all_sent = 0;
	while (all_sent < n) {
		int sent;
		LIBUSB(libusb_interrupt_transfer(dh, WRITE_EP, (uint8_t *)data + all_sent, n - all_sent, &sent, 100));
		all_sent += sent;
		assert(sent == n);
	}*/
	int sent;
	LIBUSB(libusb_bulk_transfer(dh, WRITE_EP, (uint8_t*)data, n, &sent, 100));
	assert(sent == n);
	return 0;
}

int read(libusb_device_handle *dh, void *data, int n) {
	assert(data);
	int recvd = 0;
	LIBUSB(libusb_bulk_transfer(dh, READ_EP, (uint8_t*)data, n, &recvd, 0));
	return recvd;
}

void bin_print(uint64_t x) {
	for (uint64_t i = 64; i > 0; i--) printf("%d", (bool)(x & (1ull << (i - 1))));
	printf("\n");
}

uint64_t popcount(uint64_t x) {
	uint64_t pc = 0;
	for (uint64_t i = 64; i > 0; i--) pc += (bool)(x & (1ull << (i - 1)));
	return pc;
}


void _endian(void *v, int size) {
	char *c = (char *)v;
	for (int i = 0; i < size / 2; i++) {
		char tmp = c[i];
		c[i] = c[size - i - 1];
		c[size - i - 1] = tmp;
	}
}

uint8_t i_crc8(uint64_t x) {
	uint64_t poly = ((1ull << 8) | 7) << (63 - 8);
	_endian(&x, 8);
	((uint8_t *)&x)[0] = 0;
	for (uint64_t i = 63; i >= 8; i--) {
		if (x & (1ull << i)) {
			uint64_t z = -1; //0b111...1000...0
			z <<= (i >= 9) ? i - 9 : 0;
			z |= x;
			z &= x ^ poly;
			x = z;
		}
		poly >>= 1;
	}
	return x ^ 0xff;
}

static const uint8_t crc8_array[256] = {
	0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15, 0x38, 0x3F, 0x36, 0x31, 0x24, 0x23, 0x2A, 0x2D,
	0x70, 0x77, 0x7E, 0x79, 0x6C, 0x6B, 0x62, 0x65, 0x48, 0x4F, 0x46, 0x41, 0x54, 0x53, 0x5A, 0x5D,
	0xE0, 0xE7, 0xEE, 0xE9, 0xFC, 0xFB, 0xF2, 0xF5, 0xD8, 0xDF, 0xD6, 0xD1, 0xC4, 0xC3, 0xCA, 0xCD,
	0x90, 0x97, 0x9E, 0x99, 0x8C, 0x8B, 0x82, 0x85, 0xA8, 0xAF, 0xA6, 0xA1, 0xB4, 0xb3, 0xBA, 0xbD,
	0xC7, 0xC0, 0xC9, 0xCE, 0xDB, 0xDC, 0xD5, 0xD2, 0xFF, 0xF8, 0xF1, 0xF6, 0xE3, 0xE4, 0xED, 0xEA,
	0xB7, 0xB0, 0xB9, 0xBE, 0xAB, 0xAC, 0xA5, 0xA2, 0x8F, 0x88, 0x81, 0x86, 0x93, 0x94, 0x9D, 0x9A,
	0x27, 0x20, 0x29, 0x2E, 0x3B, 0x3C, 0x35, 0x32, 0x1F, 0x18, 0x11, 0x16, 0x03, 0x04, 0x0D, 0x0A,
	0x57, 0x50, 0x59, 0x5E, 0x4B, 0x4C, 0x45, 0x42, 0x6F, 0x68, 0x61, 0x66, 0x73, 0x74, 0x7D, 0x7A,
	0x89, 0x8E, 0x87, 0x80, 0x95, 0x92, 0x9B, 0x9C, 0xB1, 0xB6, 0xBF, 0xB8, 0xAD, 0xAA, 0xA3, 0xA4,
	0xF9, 0xFE, 0xF7, 0xF0, 0xE5, 0xE2, 0xEB, 0xEC, 0xC1, 0xC6, 0xCF, 0xC8, 0xDD, 0xDA, 0xD3, 0xD4,
	0x69, 0x6E, 0x67, 0x60, 0x75, 0x72, 0x7B, 0x7C, 0x51, 0x56, 0x5F, 0x58, 0x4D, 0x4A, 0x43, 0x44,
	0x19, 0x1E, 0x17, 0x10, 0x05, 0x02, 0x0B, 0x0C, 0x21, 0x26, 0x2F, 0x28, 0x3D, 0x3A, 0x33, 0x34,
	0x4E, 0x49, 0x40, 0x47, 0x52, 0x55, 0x5C, 0x5B, 0x76, 0x71, 0x78, 0x7F, 0x6A, 0x6D, 0x64, 0x63,
	0x3E, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2C, 0x2B, 0x06, 0x01, 0x08, 0x0F, 0x1A, 0x1D, 0x14, 0x13,
	0xAE, 0xA9, 0xA0, 0xA7, 0xB2, 0xB5, 0xBC, 0xBB, 0x96, 0x91, 0x98, 0x9F, 0x8A, 0x8D, 0x84, 0x83,
	0xDE, 0xD9, 0xD0, 0xD7, 0xC2, 0xC5, 0xCC, 0xCB, 0xE6, 0xE1, 0xE8, 0xEF, 0xFA, 0xFD, 0xF4, 0xF3
};
uint8_t a_crc8_array(uint8_t *ptr, int num_bytes) {
	uint8_t crc = 0x00;
	for (uint8_t i = 0; i < num_bytes; i++) {
		crc = crc8_array[crc ^ ptr[i]];
	}
	return crc ^ 0xff;
}
uint8_t a_crc8(uint64_t x) {
	return a_crc8_array((uint8_t *)&x, 7);
}




void test_crc8() {
	uint8_t i8 = 0x69;
	_endian(&i8, 1);
	assert(i8 == 0x69);
	uint16_t i16 = 0x1337;
	_endian(&i16, 2);
	assert(i16 == 0x3713);
	uint32_t i32 = 0x13374269;
	_endian(&i32, 4);
	assert(i32 == 0x69423713);

	assert(i_crc8(0x33) == 0xeb);
	assert(i_crc8(0x31) == 0x52);
	assert(i_crc8(0x4731) == 0xfb);
	assert(i_crc8(0x11111111111111ull) == 0x99);
	assert(i_crc8(0x56111111111111ull) == 0x4B);

	uint64_t frames[] = { 0x23d68ull,0x23d69ull,0x23d6aull };
	uint8_t crcs[] = { 0x52,0x8d,0xeb };
	for (int i = 0; i < 3; i++) {
		printf("0x%x\n", i_crc8(frames[i]));
		assert(i_crc8(frames[i]) == crcs[i]);
	}
}

frame make_frame(char cmd, uint8_t rw, uint32_t data) {
	frame f = { 0 };
	f.header = cmd;
	f.sc = '=';
	f.rw = rw;
	f.idata = htonl(data);
	f.crc = a_crc8(*(uint64_t*)&f);
	f.end = '&';
	return f;
}


void print_frame(frame f) {
	for (int i = 0; i < 9; i++) printf("0x%x\t", ((uint8_t *)&f)[i]);
	puts("");
}

void print_frame32(frame f) {
	for (int i = 0; i < 32; i++){
		uint8_t v=((uint8_t *)&f)[i];
		if(v<16) printf("0");
		printf("%x ",v );
	}
	puts("");
}


frame send_frame(libusb_device_handle* dh, frame f1) {
	//printf("req "); print_frame(f1);
	frame f2;
	write(dh, &f1, sizeof(frame));
	int n = read(dh, &f2, sizeof(frame));
	assert(n > 0);
	//printf("res "); print_frame(f2);
	if (f1.rw != f2.rw) {
#define CASE(name) case name: printf("send_frame error: %s cmd:%c\n",#name,f1.header);break;
		switch (f2.rw) {
			CASE(CRC_ERROR);
		default: puts("unknown error");
		}
#undef CASE
	}
	return f2;
}

inline float accel_i2f(int x) {
	return (int16_t)x / 1693.6;
}

vec3 get_accel(libusb_device_handle* dh) {
	frame f;
	vec3 accel;
	f = send_frame(dh, make_frame('h', RINT, 0));
	bin_print(f.idata);
	accel.x = accel_i2f(f.idata);
	f = send_frame(dh, make_frame('i', RINT, 0));
	bin_print(f.idata);
	accel.y = accel_i2f(f.idata);
	f = send_frame(dh, make_frame('j', RINT, 0));
	bin_print(f.idata);
	accel.z = accel_i2f(f.idata);
	return accel;
}


uint32_t get_dist(libusb_device_handle *dh) {
	frame f;
	frame f1=make_frame('k', RINT,0);
	f = send_frame(dh, f1);
	return htonl(f.idata);
}

float get_volts(libusb_device_handle *dh, bool five_volts) {
	frame f1=make_frame(five_volts ? 'u' : 'v', RINT, 0);
	frame f = send_frame(dh, f1);
	return (five_volts ? 5.0f : 3.3f) * f.idata / 4096.0f;
}

float get_led_temp(libusb_device_handle *dh,bool board1) {
	frame f;
	frame f1=make_frame((board1) ? 'U' : 'V', RFLOAT, 0);
	f = send_frame(dh, f1);
	//bin_print(f.idata);
	uint32_t x=htonl(f.idata);
	return *(float *)&x;
}


status get_status(libusb_device_handle *dh) {
	frame f1=make_frame('A', RINT, 0);
	frame f = send_frame(dh, f1);
	status s;
	s.led1_ok = f.data[3] & 1;
	s.led2_ok = f.data[3] & 2;
	s.accel_ok = f.data[3] & 4; 
	s.dist_ok = f.data[3] & 8;
	return s;
}

void set_camera_power(libusb_device_handle* dh, bool on) {
	send_frame(dh, make_frame('B', 0, (uint32_t)on));
}

void set_color_camera_power(libusb_device_handle* dh, bool on) {
	send_frame(dh, make_frame('C', 0, (uint32_t)on));
}



void print_date(date d) {
	printf("%d/%d/%d at %d:%d \n", d.month, d.day, d.year + 2022, d.hour, d.minute);
}

date get_calibration_date(libusb_device_handle * dh, bool sensor1) {
	frame f;
	frame f1=make_frame('s', RINT, (int32_t)sensor1);
	f = send_frame(dh, f1);
	date d;
	d.i = f.idata;
	_endian(&d, 4);
	return d;
}


void set_led_power(libusb_device_handle* dh, bool on) {
	if (on) send_frame(dh, make_frame('E', 0, 0));
	frame f = send_frame(dh, make_frame('D', 0, on));
}

void set_ff_led_power(libusb_device_handle* dh, bool on) {
	if (on) send_frame(dh, make_frame('D', 0, 0));
	frame f = send_frame(dh, make_frame('E', 0, on));
}

void set_ff_green_led_state(libusb_device_handle* dh, bool on) {
	frame f;
	if (on) f = send_frame(dh, make_frame('g', 0, 50));
	f = send_frame(dh, make_frame('O', 0, on));
}
void set_ff_magenta_led_state(libusb_device_handle* dh, bool on) {
	frame f;
	if (on) f = send_frame(dh, make_frame('f', 0, 50));
	f = send_frame(dh, make_frame('N', 0, on));
}


enum led_color_cmd color2cmd(enum led_color c) {
	switch (c) {
	case C_BLUE: return CC_BLUE;
	case C_GREEN: return CC_GREEN;
	case C_AMBER: return CC_AMBER;
	case C_DEEP_RED: return CC_DEEP_RED;
	case C_FAR_RED: return CC_FAR_RED;
	case C_NIR1: return CC_NIR1;
	case C_NIR2: return CC_NIR2;
	case _C_NONE: return _CC_NONE;
	case C_RED: return CC_RED;
	}
	assert(0);
	return _CC_NONE;
}



void set_led_state(libusb_device_handle* dh, enum led_color c, bool on){
	frame f2 = send_frame(dh, make_frame(color2cmd(c), 0, on));
}

void set_led_current(libusb_device_handle* dh, enum led_color c, uint8_t x) {
	enum led_color_cmd cmd = color2cmd(c);
	frame f2 = send_frame(dh, make_frame(cmd + ((cmd < CC_AMBER) ? ('X' - CC_BLUE) : ('a' - CC_RED)), 0, x));
}

void reinit_chips(libusb_device_handle* dh) {
	send_frame(dh, make_frame('A', 8, 1));
}




void raw_led_test(libusb_device_handle * dh) {
	uint8_t raw_messages[][9] = {
		// D = ? ? ? ?  � & 
		{68,61,0,0,0,0, 1,175,38},
		// X = ? ? ? ? 2 z & 
		{88,61,0,0,0,0,50,122,38},
		// F = ? ? ? ?   & 
		{70,61,0,0,0,0, 1, 22,38},
		// F = ? ? ? ? ?  & 
		{70,61,0,0,0,0, 0, 17,38},
		// D = ? ? ? ? ? � & 
		{68,61,0,0,0,0, 0,168,38},
	};
#if 0
	for (int i = 0; i < 5; i++) {
		frame req;
		memcpy(&req, raw_messages[i], 9);
		frame rsp = send_frame(dh, req);
	}
#else
	// uint8_t col=BLUE;
	// uint8_t col=RED;
	// uint8_t col=GREEN;
	// uint8_t col=AMBER;
	led_color col = C_DEEP_RED;
	//set_led_state(dh, C_ALL, 1);
	set_led_power(dh, 1);
	set_led_current(dh,col, 50);
	set_led_state(dh, col, 1);
	usleep(2.5e2*1e3);
	set_led_state(dh, col, 0);
	set_led_power(dh, 0);
#endif
}

bool poll_button(libusb_device_handle *dh) {
#ifdef DVCOM
	frame in = make_frame('W', RINT, 0);
#else
	frame in = make_frame('W', 0x1d, 0);
#endif 
	
	volatile frame out = send_frame(dh, in);
	volatile bool b = out.idata==0;
	return b;
}

void get_firmware_version(libusb_device_handle *dh, char *out, int len) {
	frame f1;
	uint8_t *buf = (uint8_t *)&f1;
	memset(buf, 0, 32);
	buf[0] = 'p';//0x70;
	buf[1] = 0x3d;
	buf[2] = 0x06;
	buf[32 - 2] = 0x41;
	buf[32 - 1] = 0x26;
	//print_frame(f1);
	frame f2 = send_frame(dh, f1);
	strncpy(out, ((char *)&f2) + 3, min(len,32-3-2));

}


void set_all_led_currents(libusb_device_handle *dh){

}

frame set_led_groups(libusb_device_handle *dh,bool white,bool visible,bool nir){
	puts(__func__);
	frame f1;
	uint8_t *buf = (uint8_t *)&f1;
	memset(buf, 0, 32);
	buf[0] = 'r';//0x70;
	buf[1] = 0x3d;
	buf[2] = 0x2a;
	buf[3] = white;
	buf[4] = visible;
	buf[5] = nir;
	buf[32 - 2] = a_crc8_array((uint8_t*)&f1,30);
	buf[32 - 1] = 0x26;
	print_frame32(f1);
	frame f2 = send_frame(dh, f1);
	return f2;
}

frame get_led_groups(libusb_device_handle *dh){
	puts(__func__);
	frame f1;
	uint8_t *buf = (uint8_t *)&f1;
	memset(buf, 0, 32);
	buf[0] = 'r';//0x70;
	buf[1] = 0x3d;
	buf[2] = 0x2b;
	buf[32 - 2] = a_crc8_array((uint8_t*)&f1,30);
	buf[32 - 1] = 0x26;
	print_frame32(f1);
	frame f2 = send_frame(dh, f1);
	return f2;
}

frame set_all_led_currents(libusb_device_handle *dh,int v){
	puts(__func__);
	frame f1;
	uint8_t *buf = (uint8_t *)&f1;
	memset(buf, 0, 32);
	buf[0] = 'r';//0x70;
	buf[1] = 0x3d;
	buf[2] = 0x2c;
	for(int i=0;i<13;i++) buf[3+i]=v;
	buf[32 - 2] = a_crc8_array((uint8_t*)&f1,30);
	buf[32 - 1] = 0x26;
	print_frame32(f1);
	frame f2 = send_frame(dh, f1);
	return f2;
}

frame get_all_led_currents(libusb_device_handle *dh){
	puts(__func__);
	frame f1;
	uint8_t *buf = (uint8_t *)&f1;
	memset(buf, 0, 32);
	buf[0] = 'r';//0x70;
	buf[1] = 0x3d;
	buf[2] = 0x2d;
	buf[32 - 2] = a_crc8_array((uint8_t*)&f1,30);
	buf[32 - 1] = 0x26;
	print_frame32(f1);
	frame f2 = send_frame(dh, f1);
	return f2;
}

_float2x3 read_gyroscope(libusb_device_handle *dh){
	// puts(__func__);
	frame f1;
	uint8_t *buf = (uint8_t *)&f1;
	memset(buf, 0, 32);
	buf[0] = 'r';//0x70;
	buf[1] = 0x3d;
	buf[2] = 0x35;
	buf[32 - 2] = a_crc8_array((uint8_t*)&f1,30);
	buf[32 - 1] = 0x26;
	frame f2 = send_frame(dh, f1);
	// print_frame32(f2);

	struct __attribute__((packed)) packet_t{
		uint8_t header[3];
		int16_t gx,gy,gz;
		int16_t ax,ay,az;
	};
	packet_t p=*(packet_t*)&f2;

	float gx=pi*((p.gx)/131.068)/180;
	float gy=pi*((p.gy)/131.068)/180;
	float gz=pi*((p.gz)/131.068)/180;

	float ax=p.ax/1693.6;
	float ay=p.ay/1693.6;
	float az=p.az/1693.6;

	// printf("x:%f\ty:%f\tz:%f\n",x,y,z);
	// printf("x:%x\ty:%x\tz:%x\n",p.x,p.y,p.z);
	// println(x,y,z);

	return _float2x3(_float3(gx,gy,gz),_float3(ax,ay,az));
}

void stm_disconnect(libusb_device_handle *dh) {
	libusb_close(dh);
}

libusb_device_handle *stm_connect(){
	test_crc8();
	libusb_device **list = NULL;
	LIBUSB(libusb_init(NULL));
	// LIBUSB(libusb_set_option(NULL, LIBUSB_OPTION_LOG_LEVEL, 3));
	int num_devs = 0;
	LIBUSB(num_devs = libusb_get_device_list(NULL, &list));
	assert(num_devs > 0);
	for (int i = 0; i < num_devs; i++) {
		struct libusb_device_descriptor dd;
		LIBUSB(libusb_get_device_descriptor(list[i], &dd));
		libusb_device_handle *dh = NULL;

		if (libusb_open(list[i], &dh) < 0) continue;

		/*char str[256];
		LIBUSB(libusb_get_string_descriptor_ascii(dh, dd.iManufacturer, (uint8_t *)str, sizeof(str)));
		printf("iManufacturer: %s\n", str);
		LIBUSB(libusb_get_string_descriptor_ascii(dh, dd.iProduct, (uint8_t *)str, sizeof(str)));
		printf("iProduct: %s\n", str);*/

		//if (strcmp("STM32F413_Bulk", str) == 0) {
		if (dd.idVendor == 1155 && dd.idProduct == 22288) {
			puts("match");
			struct libusb_config_descriptor *config = NULL;
			LIBUSB(libusb_get_active_config_descriptor(list[i], &config));

			LIBUSB(libusb_claim_interface(dh, 0));

			return dh;
		}
	}
	// THROW("cannot connect to microcontroller");
    assert(0);
	return NULL;
}