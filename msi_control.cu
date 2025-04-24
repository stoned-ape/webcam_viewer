#include "common.h"
#include "cv_debayer.h"
#include "kernels.h"
#include "zoomer.h"
#include "camera.h"
#include "usb_api.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/opencv.hpp>

#define DVCOM

#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h> 
#include <unistd.h>
#include <pthread.h>


#undef max
#undef min
#include <atomic>
// #include <thread>


#include <libusb-1.0/libusb.h>


#define PRINT(val,type) printf("%s:\t" type "\n",#val,val);

#define MAX_PATH 256


inline float _map(float t, float t0, float t1, float s0, float s1) {
	return s0 + (s1 - s0) * (t - t0) / (t1 - t0);
}


struct msi_exception{
	const char *file;
	const char *func;
	const char *message;
	int line;
	void print() {
		fprintf(
			stderr,
			"exception thrown (%s) on line %d of file %s in function %s\n",
			message, line, file, func
		);
	}
	int print_str(char *buf,size_t buf_size) {
		return snprintf(
			buf,buf_size,
			"exception thrown (%s) on line %d of file %s in function %s\n",
			message, line, file, func
		);
	}
};

msi_exception global_exception = { NULL,NULL,NULL,0 };

#define THROW(message) {throw global_exception=msi_exception{__FILE__,__func__,message,__LINE__};}

#define exc_assert(e) {if(!(e)) THROW("assertion failed: " #e)}


struct rgba8 {
	uint8_t b, g, r, a;
};

struct __attribute__((packed)) bmp_header {
	uint16_t magic;
	uint32_t file_size;
	uint32_t app;
	uint32_t offset;
	uint32_t info_size;
	int32_t width;
	int32_t height;
	uint16_t planes;
	uint16_t bits_per_pix;
	uint32_t comp;
	uint32_t comp_size;
	uint32_t xres;
	uint32_t yres;
	uint32_t cols_used;
	uint32_t imp_cols;
};

void make_bmp(void* buf, int w, int h, const char* fname) {
	// puts("make_bmp()");
	bmp_header head;
	assert(offsetof(bmp_header, file_size) == 2);
	assert(sizeof(head) == 54);
	memset(&head, 0, sizeof(head));

	head.magic = 0x4d42;
	head.offset = sizeof(bmp_header);
	head.info_size = 40;
	head.width = w;
	head.height = h;
	head.planes = 1;
	head.bits_per_pix = 32;
	head.comp_size = head.width * head.height * head.bits_per_pix / 8;
	head.file_size = sizeof(bmp_header) + head.comp_size;

	FILE* bmp = fopen(fname, "wb");
	assert(fwrite(&head, 1, sizeof(head), bmp) == sizeof(head));
	assert(fwrite(buf, 1, head.comp_size, bmp) == head.comp_size);
	fclose(bmp);
}


void bayer(const uint8_t* dbr, uint8_t* byr, int w, int h) {
	const int w2 = w << 1;
	for (int j = 0; j < h; j++) {
		int j2 = j << 1;
		for (int i = 0; i < w; i++) {
			int i2 = i << 1;
			int dbr_idx = (i + w * j) << 2;
			byr[(i2 + 0) + w2 * (j2 + 0)] = dbr[dbr_idx + 2];
			byr[(i2 + 1) + w2 * (j2 + 0)] = dbr[dbr_idx + 1];
			byr[(i2 + 0) + w2 * (j2 + 1)] = dbr[dbr_idx + 3];
			byr[(i2 + 1) + w2 * (j2 + 1)] = dbr[dbr_idx + 0];
		}
	}
}


void debayer(const uint8_t* byr, uint8_t* dbr, int w, int h, bool alpha=false) {
	const int w2 = w << 1;
	for (int j = 0; j < h; j++) {
		int j2 = j << 1;
		for (int i = 0; i < w; i++) {
			int i2 = i << 1;
			int dbr_idx = (i + w * j) << 2;
			//dbr_idx+0=blue
			//dbr_idx+1=green
			//dbr_idx+2=red
			//dbr_idx+3=unused
			dbr[dbr_idx + 2] = byr[(i2 + 0) + w2 * (j2 + 0)];
			dbr[dbr_idx + 1] = byr[(i2 + 1) + w2 * (j2 + 0)];
			dbr[dbr_idx + 3] = byr[(i2 + 0) + w2 * (j2 + 1)];
			dbr[dbr_idx + 0] = byr[(i2 + 1) + w2 * (j2 + 1)];
		}
	}
	if (alpha) for (int i = 3; i < w * h * 4; i += 4) dbr[i] = 255;
	
}

// void cv_debayer(/*const*/ uint8_t* byr, uint8_t* dbr, int w, int h) {
// 	cv::Mat img1(h, w, CV_8UC1, byr, w);
// 	//static 
// 	cv::Mat img2(h, w, CV_8UC4, dbr, 0);
// 	const void* old = img2.datastart;
// 	assert(dbr == img2.datastart);
// 	cv::cvtColor(img1, img2, cv::COLOR_BayerRG2RGBA, 4);
// 	assert(old == img2.datastart);
// 	assert(byr == img1.datastart);
// 	cv::cvtColor(img2, img2, cv::COLOR_BGRA2RGBA, 4);
// }


void test_debayer() {
	const uint8_t byr[4 * 4] = {
		'R' | 1 | 16,'G' | 2 | 16,'R' | 4 | 16,'G' | 8 | 16,
		'G' | 1 | 32,'B' | 2 | 32,'G' | 4 | 32,'B' | 8 | 32,
		'R' | 1 | 64,'G' | 2 | 64,'R' | 4 | 64,'G' | 8 | 64,
		'G' | 1 | 128,'B' | 2 | 128,'G' | 4 | 128,'B' | 8 | 128,
	};
	/*const uint8_t expected_dbr[4*4]={
		'R'|1| 16,'G'|2| 16,'B'|2| 32,'G'|1| 32, 'R'|4| 16,'G'|8| 16,'B'|8| 32,'G'|4| 32,
		'R'|1| 64,'G'|2| 64,'B'|2|128,'G'|1|128, 'R'|4| 64,'G'|8| 64,'B'|8|128,'G'|4|128,
	};*/
	const uint8_t expected_dbr[4 * 4] = {
		'B' | 2 | 32,'G' | 2 | 16,'R' | 1 | 16,'G' | 1 | 32, 'B' | 8 | 32,'G' | 8 | 16,'R' | 4 | 16,'G' | 4 | 32,
		'B' | 2 | 128,'G' | 2 | 64,'R' | 1 | 64,'G' | 1 | 128, 'B' | 8 | 128,'G' | 8 | 64,'R' | 4 | 64,'G' | 4 | 128,
	};

	uint8_t dbr[4 * 4];
	debayer(byr, dbr, 2, 2);
	assert(0 == memcmp(dbr, expected_dbr, 4 * 4));
	uint8_t new_byr[4 * 4];
	bayer(dbr, new_byr, 2, 2);
	assert(0 == memcmp(byr, new_byr, 4 * 4));
#if 0
	int sz = 16 * 3;
	uint8_t big_byr[sz * sz];
	uint8_t big_dbr[sz * sz];
	uint8_t* a = big_byr, * b = big_dbr;
	float t = itime();
	for (int i = 0; i < 1000000; i++) {
		debayer(a, b, sz / 2, sz / 2);
		__auto_type* tmp = a;
		a = b;
		b = tmp;
	}
	PRINT(itime() - t, "%f s");
	exit(0);
#endif
}

extern "C" void test_throw() {
	try {
		THROW("test");
	} catch (msi_exception e) {
		auto *ep = &e;
		PRINT(ep, "%p");
		assert(0);
		// RaiseException(0, 0, 1, (const ULONG_PTR *)ep);
	}
}


struct exposures_t {
	double data[8];
};


struct msi_control { 
	static constexpr int buf_count = 3;//camera::buf_count;
	static constexpr int w = 1344;//camera::w;//2448;// 3840;
	static constexpr int h = 1344;//camera::h;//2048;// 2160;
	cam_data_t cams[4];
	uint16_t debayered_images[4][buf_count][h][w][4];
	double times[buf_count];
	double exposure[4] = { 20e3,20e3,20e3,20e3 };
	libusb_device_handle *dh = NULL;
	char file_names[4][buf_count][MAX_PATH];
	msi_exception exc;
	bool cons_failed = false;
	float led_temps[4];
	double cam_temps[4];
	uint8_t led_currents[7] = { 82,87 ,203,48 ,94 ,98 ,101 };
	vec3 acc[4];
	uint32_t dist;
	double ret_exps;
    struct gpiod_chip *chip=NULL;
    struct gpiod_line *line=NULL;
	msi_control() {
		try {
			dh = stm_connect();
		} catch (msi_exception e) {
			exc = e;
			cons_failed = true;
		}
	}
	~msi_control() {
		if(!cons_failed) stm_disconnect(dh);
	}
	void init() {
		if (cons_failed) throw exc;
#ifdef DVCOM
#ifdef MICHAEL
		michael_lib.OpenUSB();
		michael_lib.set_camera_power(true);
#else
		set_camera_power(dh, true);
#endif
#endif
		
		chip=gpiod_chip_open_by_name("gpiochip0");
	    assert(chip);
	    line=gpiod_chip_get_line(chip, 108); //msi cam
        assert(line);
        gpiod_line_set_value(line,1);
        gpiod_line_request_output(line, "example", 0);
    

		static struct{char name[16];char card[32];} devs_info[10]={0};
	    int devs_count=0;
	    int devs_select=3;

	    #define MAX_DEVS 10
	    // (sizeof(devs_info)/sizeof(devs_info[0]))

	    DIR *dev_dir=opendir("/dev");
	    assert(dev_dir);
	    struct dirent *entry;
	    while((entry=readdir(dev_dir))!=NULL){
	        if(0==memcmp(entry->d_name,"video",5)){
	           if(devs_count>=MAX_DEVS) break;
	           memcpy(devs_info[devs_count].name,"/dev/",5);
	           strcpy(devs_info[devs_count].name+5,entry->d_name);
	           devs_count++;
	        }
	    }
	    for(int i=0;i<devs_count;i++){
	        int fd=SYSCALL(open(devs_info[i].name,O_RDWR));
	        if(fd>0){
	            struct v4l2_capability cap={0};
	            SYSCALL(ioctl(fd,VIDIOC_QUERYCAP,&cap));
	            memcpy(devs_info[i].card,cap.card,32);
	            SYSCALL(close(fd));
	        }
	        puts(devs_info[i].name);
	    }
	    closedir(dev_dir);
	    assert(devs_count>0);
		for(int i=0,j=0;i<devs_count,j<4;i++){
			cams[j]=init_cam(devs_info[i].name,false,w,h);

			assert((cams[j].valid));
			assert(cams[j].w==w);
			assert(cams[j].h==h);
			if(cams[j].valid) j++;
			else deinit_cam(&cams[j]);
		}

		for(int j=0;j<4;j++){
			set_trigger_mode(cams[j].fd,true,false);
			// dequeue_all(cams[j].fd);

		}


	}
	void deinit() {
		for(int i=0;i<4;i++) deinit_cam(&cams[i]);
		set_camera_power(dh, false);
	}
	void await_exp_started() {
		// for (int j = 0; j < 4; j++) while (!cams[j].exp_started);
		// for (int j = 0; j < 4; j++) cams[j].exp_started = false;
	}
	void await_exp_finished() {
		// for (int j = 0; j < 4; j++) while (!cams[j].exp_finished);
		// for (int j = 0; j < 4; j++) cams[j].exp_finished = false;
	}
	void set_exposure(double exp0, double exp1, double exp2, double exp3) {
		exposure[0] = exp0;
		exposure[1] = exp1;
		exposure[2] = exp2;
		exposure[3] = exp3;
		for (int j = 0; j < 4; j++) set_exposure_ms(cams[j].fd,exposure[j]);
		// for (int j = 0; j < 4; j++)	EBUS(cams[j].exp_auto->SetValue(0));
		// for (int j = 0; j < 4; j++)	EBUS(cams[j].exp_mode->SetValue(1));
		// for (int j = 0; j < 4; j++) cams[j].exp_time->SetValue(exposure[j]); // 8 ms
	}

	void thread_debayer_func(int cam_idx,int n) {
		for (int i = 0; i < n; i++) {
			cam_data_t cd=cams[cam_idx];
			assert(cd.raw_buf);
			memcpy(cd.raw_buf,cd.buf_ptrs[i],cd.raw_buf_size);
			int block_size=256;
			int num_blocks=(cd.w*cd.h+block_size-1)/block_size;


			bayer_to_rgb<<<num_blocks,block_size>>>(cd.raw_buf,cd.draw_buf,w,h);
			CUDA(cudaDeviceSynchronize());
			memcpy(debayered_images[cam_idx][i],cd.draw_buf,w*h*4*2);
			// cv_bayer_to_rgb((uint8_t*)cams[cam_idx].buf_ptrs[i], (uint8_t*)debayered_images[cam_idx][i], w, h);
		}
	};
	void acquire() {
		// PvResult res;

		// PvBuffer *buf = NULL;
		// PvResult opres;

		acc[0] = get_accel(dh);

		set_exposure(exposure[0], exposure[1], exposure[2], exposure[3]);
		
		// for (int j = 0; j < 4; j++) res = EBUS(cams[j].device->StreamEnable()); // 7...20 ms

		for (int i = 0; i < buf_count; i++) {
			for (int j = 0; j < 4; j++) enqueue_buf(cams[j].fd,i);
		}

		//set_led_power(dh, true);
		set_led_power(dh, 1);

		gpiod_line_set_value(line,1);

		float t0 = itime();
		for (int j = 0; j < 4; j++) start_stream(cams[j].fd);//res = EBUS(cams[j].start->Execute()); // 1 ms
		assert(buf_count == 3);
		const int seq[4] = {0,0,5,7};
		for (int i = 0; i < buf_count; i++) {
			float t0 = itime();

			
			for (int k = seq[i]; k < seq[i+1]; k++) {
				auto c = (led_color)k;
				set_led_current(dh, c, led_currents[k]);
				set_led_state(dh, c, 1);
			}
			

			// for (int j = 0; j < 4; j++) cams[j].fire->Execute();
			// await_exp_started();
			// await_exp_finished();
			//Sleep(20);

			gpiod_line_set_value(line,0);
            usleep(1e5);
            gpiod_line_set_value(line,1);

			

			for (int k = seq[i]; k < seq[i + 1]; k++) {
				auto c = (led_color)k;
				set_led_state(dh, c, 0);
			}

			float t1 = itime();
			PRINT((t1 - t0) * 1e6, "%f us");
		}
		float t1 = itime();
		PRINT(t1 - t0, "%f s total");
		for (int j = 0; j < 4; j++) stop_stream(cams[j].fd);//EBUS(cams[j].stop->Execute()); // 1...36 ms

		//set_led_state(dh, C_ALL, 0);
		set_led_power(dh, 0);


		// for (int j = 0; j < 4; j++) {
		// 	cams[j].writer.Store(cams[j].device, "device");
		// 	cams[j].writer.Store(cams[j].stream, "stream");
		// }
		// for (int j = 0; j < 4; j++) EBUS(cams[j].device->StreamDisable()); // 14 ms

		{
			float t0 = itime();
#if 0
			for (int j = 0; j < 2; j++) {
				for (int i = 0; i < 8; i++) {
					cv_debayer((uint8_t*)cams[j].buffers[i].GetDataPointer(), (uint8_t*)debayered_images[j][i], w, h);
				}
			}
#else 

			for (int j = 0; j < 4; j++) thread_debayer_func(j,buf_count);
#endif
			float t1 = itime();
			PRINT((t1 - t0) * 1e3, "debayering took %f ms");
		}

		acc[1] = get_accel(dh);
		dist = get_dist(dh);
		//led_temps[0] = get_led_temp(dh, 0);
		led_temps[1] = get_led_temp(dh, 1);

		// for (int j = 0; j < 4; j++) {
		// 	((PvGenFloat *)cams[j].dev_params->Get("DeviceTemperature"))->GetValue(cam_temps[j]);
		// }


	}
	void acquire_once_per_cam() {
		set_exposure(exposure[0], exposure[1], exposure[2], exposure[3]);

	
		gpiod_line_set_value(line,1);

		for (int j = 0; j < 4; j++) enqueue_buf(cams[j].fd,0);

		for (int j = 0; j < 4; j++) start_stream(cams[j].fd);

		gpiod_line_set_value(line,0);
		usleep(1e5);
		gpiod_line_set_value(line,1);

		for (int j = 0; j < 4; j++) stop_stream(cams[j].fd);

		// for (int j = 0; j < 4; j++) dequeue_buf(cams[j].fd);

	
		for (int j = 0; j < 4; j++) thread_debayer_func(j,1);
		
	}

	void acquire_once(int cam_idx) {
		/*
		PvResult res;
		PvResult opres;
		PvBuffer *buf = NULL;

		//set_exposure(exposure);
		set_exposure(exposure[0], exposure[1], exposure[2], exposure[3]);

		res = EBUS(cams[cam_idx].device->StreamEnable()); // 7...20 ms


		cams[cam_idx].stream->QueueBuffer(&cams[cam_idx].buffers[0]); // 3 ms

		res = EBUS(cams[cam_idx].start->Execute()); // 1 ms


		float t0 = itime();
		cams[cam_idx].fire->Execute();
		while (!cams[cam_idx].exp_finished);
		cams[cam_idx].exp_finished = false;
		usleep(20*1e3);
		float t1 = itime();
		PRINT((t1 - t0) * 1e6, "%f us");


		EBUS(cams[cam_idx].stop->Execute()); // 1...36 ms

		
		res = cams[cam_idx].stream->RetrieveBuffer(&buf, &opres);//,exposures[i]+1000));
		EBUS(res);
		EBUS(opres);
		exc_assert(buf == &cams[cam_idx].buffers[0]);
		
		EBUS(cams[cam_idx].device->StreamDisable()); // 14 ms

		//debayer((uint8_t*)cams[cam_idx].buffers[0].GetDataPointer(), (uint8_t*)debayered_images[cam_idx][0], w / 2, h / 2, true);
		cv_debayer((uint8_t*)cams[cam_idx].buffers[0].GetDataPointer(), (uint8_t*)debayered_images[cam_idx][0], w, h);
		*/
	}

	void save_buffer(const char *folder,int cam_idx,int seq) {
		
		//LEDID_CAMGUID_year-month-day_hour.min.sec.tiff
		time_t time_raw_format;
		struct tm *ptr_time;
		static char time_str[50];
		time(&time_raw_format);
		ptr_time = localtime(&time_raw_format);
		strftime(time_str, 50, "%Y-%m-%d_%H.%M.%S", ptr_time);
		int led_class;
		if (seq == 0) led_class = 0;
		else if (seq < 6) led_class = 1;
		else led_class = 2;
		sprintf(file_names[cam_idx][seq], "%s/%d_%d_%d_%s.tiff", 
			folder, led_class, seq, cam_idx, time_str);

		cv::Mat mat2(h, w, CV_16UC4, (uint8_t *)(debayered_images[cam_idx][seq]), 0);
		cv::imwrite(file_names[cam_idx][seq], mat2);

		
	}
	void save_images(const char *folder = NULL) {
		puts(__func__);
		if (folder == NULL) folder = ".";
		// for (int i = 0; i < 4; i++) {
		// 	static char xmlfname[MAX_PATH];
		// 	sprintf(xmlfname, "%s/camera_%d_params.xml",folder,i);
		// 	cams[i].writer.Save(xmlfname);
		// }
		double t0 = itime();
#if 0
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < buf_count; j++) {
				save_buffer(folder, i, j);
			}
		}
#else
		struct args_t{
			msi_control *_this;
			int cam_idx;
			int buf_count;
			const char *folder;
		};
		args_t args[4];
		auto func=[](void *v)->void*{
			args_t args=*(args_t*)v;
			for (int i=0;i<args.buf_count;i++) {
				args._this->save_buffer(args.folder,args.cam_idx,i);
			}
		};
		// std::thread *threads[3];
		pthread_t threads[4];
		for (uint32_t i = 0; i < 4; i++){
			args[i]._this=this;
			args[i].cam_idx=i;
			args[i].buf_count=buf_count;
			args[i].folder=folder;
			pthread_create(&threads[i],NULL,func,(void*)&args[i]);
			//threads[i] = new std::thread(func, i);
		}

		for (uint32_t i = 0; i < 4; i++) {
			// threads[i]->join();
			// delete threads[i];
			pthread_join(threads[i],NULL);
		}
#endif
		double t1 = itime();
		println("save_images took: ", t1 - t0, " s");

		char acq_data_path[MAX_PATH];
		sprintf(acq_data_path, "%s/AcquisitionData.txt", folder);
		FILE *f=fopen(acq_data_path, "w");
		fprintf(f, "Temperatures\n\tled1: %f C\n\tled2: %f C\n\tcam1: %f C\n\tcam2: %f C\n\n",led_temps[0],led_temps[1],cam_temps[0],cam_temps[1]);

		fprintf(f, "Exposure\n\t%f us\n\n", exposure);

		fprintf(f, "Currents\n");
		for (int i = 0; i < 7; i++) fprintf(f, "\t%u\n", led_currents[i]);
		fprintf(f,"\n");
		// fprintf(f,"Timestamps\n");
		// for (int i = 0; i < 2; i++) {
		// 	fprintf(f, "\t%s : ", (const char *)cams[i].guid);
		// 	for (int j = 0; j < buf_count; j++) {
		// 		fprintf(f, "%llu, ", cams[i].buffers[j].GetTimestamp());
		// 	}
		// 	fprintf(f, "\n");
		// }
		// fprintf(f,"\n");
		for (int i = 0; i < 2; i++) {
			fprintf(f, "Orientation %s\n\tX : %f m/s^2\n\tY : %f m/s^2\n\tZ : %f m/s^2\n\n",i?"After":"Before", acc[i].x, acc[i].y, acc[i].z);
		}
		fprintf(f, "Working distance: %u mm\n\n", dist);
		fclose(f);
	}
	void show_images(uint16_t bitfield){
		uint32_t pow2=1;
		for(int i=0;i<16;i++){
			if (pow2 & bitfield) {
				static char cmd_buf[MAX_PATH];
				sprintf(cmd_buf, "\"%s\"", file_names[i % 2][i / 2]);
				system(cmd_buf);
			}
			pow2<<=1;
		}
	}
	void test_image_color() {
		struct drgb {
			double r, g, b;
		};
		drgb color_sums[2][buf_count] = {0};
		auto get_color_sum = [](uint8_t *ptr) ->drgb {
			drgb sum = { 0,0,0 };
			for (int j = 0; j < h; j++) {
				int jm = j % 2;
				for (int i = 0; i < w; i++) {
					uint8_t val = ptr[i + j * w];
					int im = i % 2;
					if (im != jm) sum.g += val;
					else if (im) sum.r += val;
					else sum.b += val;
				}
			}
			return sum;
		};
		for (int j = 0; j < 2; j++) {
			for (int i = 0; i < buf_count; i++) {
				color_sums[j][i] = get_color_sum((uint8_t*)cams[j].buf_ptrs[i]);
			}
		}
		//{ NO_LED ,BLUE,GREEN,AMBER,DEEP_RED,FAR_RED,NIR1,NIR2 }
		for (int j = 0; j < 2; j++) {
			//blue
			exc_assert(color_sums[j][0].b < color_sums[j][1].b);
			//green
			exc_assert(color_sums[j][0].g < color_sums[j][2].g);
			//amber
			exc_assert(color_sums[j][0].g < color_sums[j][3].g);
			exc_assert(color_sums[j][0].r < color_sums[j][3].r);
			//deep red
			exc_assert(color_sums[j][0].r < color_sums[j][4].r);
			//far red
			exc_assert(color_sums[j][0].r < color_sums[j][5].r);
			
			//NIR1
			exc_assert(color_sums[j][0].r < color_sums[j][6].r);
			exc_assert(color_sums[j][0].b < color_sums[j][6].b);
			//NIR2
			exc_assert(color_sums[j][0].r < color_sums[j][7].r);
			exc_assert(color_sums[j][0].b < color_sums[j][7].b);

		}


	}
	static double get_grey_sum(uint8_t *ptr) {
		double sum = 0;
		for (int j = 0; j < w * h; j++) sum += ptr[j];
		return sum;
	};
	void test_exposure() {
		double sums[2][2][8];
		constexpr double exps[2] = { 20e3,60e3 };
		init();
		for (int k = 0; k < 2; k++) {
			exposure[0] = exps[k];
			exposure[1] = exps[k];
			acquire();
			//save_images("D:\\images");
			for (int j = 0; j < 2; j++) {
				for (int i = 0; i < 8; i++) {
					sums[k][j][i] = get_grey_sum((uint8_t*)cams[j].buf_ptrs[i]);
				}
			}
		}
		deinit();

		static_assert(exps[1] > exps[0],"");
		
		for (int j = 0; j < 2; j++) {
			for (int i = 0; i < 8; i++) {
				exc_assert(sums[1][j][i] > sums[0][j][i]);
			}
		}
	}
	void test_current() {
		double sums[2][2][7];
		constexpr uint8_t currents[2][7] = { 
			// {20,30,80,20,100,20,80},
			// {70,60,20,80,10,80,20},
			{200,20,200,20,200,20,200},
			{20,200,20,200,20,200,20},
		};
		init();
		for (int k = 0; k < 2; k++) {
			
			memcpy(led_currents, currents[k], 7);
			exposure[0] = 20e3;
			exposure[1] = 20e3;

			acquire();
			//save_images("D:\\images");
			for (int j = 0; j < 2; j++) {
				for (int i = 0; i < 7; i++) {
					sums[k][j][i] = get_grey_sum((uint8_t*)cams[j].buf_ptrs[i+1])/(w*h);
				}
			}
		}
		deinit();

		for (int i = 0; i < 7; i++) {
			bool b1 = currents[1][i] > currents[0][i];
			for (int j = 0; j < 2; j++) {
				bool b2 = sums[1][j][i] > sums[0][j][i];
				exc_assert(b1 == b2);
			}
		}
	}
	void test_micro() {
		vec3 a = get_accel(dh);
		exc_assert(fabsf(9.81f - sqrtf(a.x * a.x + a.y * a.y + a.z * a.z)) < .5);
		auto d = get_dist(dh);
		exc_assert(d >= 40);
		exc_assert(d <= 4000);
	}
	void run_all_tests() {
		test_micro();
		init();
		exposure[0] = 20e3;
		exposure[1] = 20e3;
		acquire();
		deinit();
		test_image_color();
		test_exposure();
		test_current();
	}
};



#if 0
static msi_control global_msi;

extern "C"  void ColorCameraOn() {
	set_color_camera_power(global_msi.dh, 1);
}
extern "C"  void ColorCameraOff() {
	set_color_camera_power(global_msi.dh, 0);
}

extern "C"  void SnapshotCamerasOn() {
	try {
		global_msi.init();
	} catch (msi_exception e) {
		e.print();
		throw e;
	}
}
extern "C"  void SnapshotCamerasOff() {
	global_msi.deinit();
}
extern "C"  void AcquireMultispectralImages() {
	global_msi.acquire();
}
extern "C"  void AcquireOncePerCamera() {
	global_msi.acquire_once_per_cam();
}
extern "C"  void AcquireOnce(int cam_idx) {
	global_msi.acquire_once(cam_idx);
}
extern "C" 	void SaveImages(const char *folder) {
	global_msi.save_images(folder);
}
extern "C" 	void SaveOncePerCamera(const char *folder) {
	global_msi.save_buffer(folder,0, 0);
	global_msi.save_buffer(folder,1, 0);
}
extern "C" 	void SaveOnce(const char *folder,int cam_idx) {
	global_msi.save_buffer(folder, cam_idx, 0);
}
extern "C" 	uint32_t GetDistance() {
	return get_dist(global_msi.dh);
}
extern "C" 	bool PollButton() {
	return poll_button(global_msi.dh);
}
extern "C" 	void SetExposureTime(double exposure_time0_us,double exposure_time1_us) {
	global_msi.set_exposure(exposure_time0_us, exposure_time1_us,exposure_time0_us, exposure_time1_us);
}
extern "C" 	double GetExposureTime(int cam_idx) {
	// double x;
	// global_msi.cams[cam_idx].exp_time->GetValue(x);
	return 0;
}
extern "C" 	void SetLedCurrents(uint8_t *currents) {
	memcpy(global_msi.led_currents, currents, 7);
}
extern "C" 	void GetLedCurrents(uint8_t *currents) {
	memcpy(currents, global_msi.led_currents, 7);
}

extern "C"  int GetLastException(char *buf,size_t buf_size) {
	return global_exception.print_str(buf, buf_size);
}
extern "C"  void SetLedState(char color, bool on) {
	set_led_state(global_msi.dh, (led_color)color, on);
}
extern "C"  void SetLedCurrent(char color,uint8_t current) {
	set_led_current(global_msi.dh,(led_color)color, current);
}
extern "C"  int64_t GetGUID(int cam_idx) {
	// auto *s=(const char *)global_msi.cams[cam_idx].guid;
	// int64_t val=strtoll(s, NULL, 16);
	return 0;//val;
}
extern "C"  uint8_t *GetBufferPtr() {
	return (uint8_t *)global_msi.debayered_images;
}
extern "C"  void GetAccel(float v[3]) {
	vec3 a=get_accel(global_msi.dh);
	memcpy(v, &a, sizeof(vec3));
}
extern "C"  int8_t GetBatteryPercentage() {
	return 0;// get_battery_percentage(global_msi.dh);
}

extern "C"  float GetBoardTemp() {
	return get_led_temp(global_msi.dh,1);
}
extern "C"  float GetCameraTemp(int cam_idx) {
	// double temp;
	// ((PvGenFloat *)global_msi.cams[cam_idx].dev_params->Get("DeviceTemperature"))->GetValue(temp);
	return 0;// temp;
}

extern "C"  void SetFFLedState(bool on) {
	set_ff_led_power(global_msi.dh, on);
}
extern "C"  void SetFFGreenLetState(bool on) {
	set_ff_green_led_state(global_msi.dh, on);
}
extern "C"  void SetFFMagentaLetState(bool on) {
	set_ff_magenta_led_state(global_msi.dh, on);
}
#endif


int main() {
	puts(__func__);
	test_debayer();
	bool exc = false;


	libusb_device_handle *dh=stm_connect();

	while(1){

		float d=get_dist(dh);
		
		println(d);
	}


	
	// set_led_power(global_msi.dh, 1);

	// set_led_state(global_msi.dh,C_BLUE,1);
	// usleep(1e6);
	// set_led_state(global_msi.dh,C_BLUE,0);

	// frame f;
	// f=set_all_led_currents(global_msi.dh,255);
	// f=get_all_led_currents(global_msi.dh);
	// print_frame32(f);

	// f=get_led_groups(global_msi.dh);
	// print_frame32(f);
	// f=set_led_groups(global_msi.dh,0,1,0);
	// while(1){

	// 	read_gyroscope(global_msi.dh);
	// 	usleep(3e4);
	// }


	// usleep(1e6);
	// f=set_led_groups(global_msi.dh,0,0,0);

	try {
		//reinit_chips(global_msi.dh);
		
		//global_msi.run_all_tests();
		//global_msi.test_current();
		//global_msi.test_exposure();

		// global_msi.init();
		//SetExposureTime(20e3, 20e3);



		// global_msi.acquire_once_per_cam();
		// global_msi.deinit();
		// global_msi.save_images("images");

		// set_led_state(global_msi.dh, C_ALL, 1);
		// set_led_state(global_msi.dh, C_GREEN, 1);
		// set_led_state(global_msi.dh, C_GREEN, 0);
		// set_led_state(global_msi.dh, C_ALL, 0);




	} catch (msi_exception e) {
		e.print();
		assert(1 == 0);
	}

	puts("done");
}



