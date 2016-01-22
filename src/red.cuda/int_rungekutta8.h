#pragma once

#include <vector>

#include "integrator.h"
#include "pp_disk.h"

#include "red_type.h"

class rungekutta8 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static var_t bh[];
	static ttt_t c[];

	rungekutta8(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev);
	~rungekutta8();

	ttt_t step();

private:
	void cpu_calc_ytemp_for_f1(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, var_t a10);
	void cpu_calc_ytemp_for_f2(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f1, var_t a20, var_t a21);
	void cpu_calc_ytemp_for_f3(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, var_t a30, var_t a32);
	void cpu_calc_ytemp_for_f4(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, const var_t *f3, var_t a40, var_t a42, var_t a43);
	void cpu_calc_ytemp_for_f5(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, var_t a50, var_t a53, var_t a54);
	void cpu_calc_ytemp_for_f6(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, var_t a60, var_t a63, var_t a64, var_t a65);
	void cpu_calc_ytemp_for_f7(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f4, const var_t *f5, const var_t *f6, var_t a70, var_t a74, var_t a75, var_t a76);
	void cpu_calc_ytemp_for_f8(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, var_t a80, var_t a83, var_t a84, var_t a85, var_t a86, var_t a87);
	void cpu_calc_ytemp_for_f9(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, var_t a90, var_t a93, var_t a94, var_t a95, var_t a96, var_t a97, var_t a98);
	void cpu_calc_ytemp_for_f10(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a100, var_t a103, var_t a104, var_t a105, var_t a106, var_t a107, var_t a108, var_t a109);
	void cpu_calc_ytemp_for_f11(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a110, var_t a115, var_t a116, var_t a117, var_t a118, var_t a119);
	void cpu_calc_ytemp_for_f12(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f11, var_t a120, var_t a123, var_t a124, var_t a125, var_t a126, var_t a127, var_t a128, var_t a129, var_t a1211);

	void cpu_calc_error(int_t n, var_t *err, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12);
	void cpu_calc_y_np1(int_t n, var_t *y_np1, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f10, var_t b0, var_t b5, var_t b6, var_t b7, var_t b8, var_t b9, var_t b10);

	void cpu_calc_ytemp_for_fr(int n_var, int r);
	void cpu_calc_y_np1(int n_var);
	void cpu_calc_error(int n_var);

	void calc_ytemp_for_fr(int n_var, int r);
	void calc_y_np1(int n_var);
	void calc_error(int n_var);
};


class c_rungekutta8 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static var_t bh[];
	static ttt_t c[];

	c_rungekutta8(pp_disk *ppd, ttt_t dt, bool adaptive, var_t tolerance, computing_device_t comp_dev);
	~c_rungekutta8();

	ttt_t step();

private:
	void call_kernel_calc_ytemp(int n_var, int r);
	void call_kernel_calc_y_np1(int n_var);
	void call_kernel_calc_error(int n_var);

	vector<vector <var4_t*> >	dydx;	//!< Holds the derivatives for the differential equations
	var4_t**                     d_dydt; //!< Vector of vectors on the device: contains a copy of the dydx vector
};
