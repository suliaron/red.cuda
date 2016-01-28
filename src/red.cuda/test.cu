#include <stdio.h>

#include "red_type.h"

using namespace std;

void test_n_objects_t()
{
	const char test_set[] = "test_n_objects_t";

	fprintf(stderr, "TEST: %s\n", test_set);

	// Test get_n_total_initial()
	{
		char test_func[] = "get_n_total_initial";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 28;
		int result = n_bodies.get_n_total_initial();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_total_playing()
	{
		char test_func[] = "get_n_total_playing";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 28;
		int result = n_bodies.get_n_total_playing();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		for (int i = 0; i < BODY_TYPE_N; i++)
		{
			n_bodies.inactive[i]++;
		}
		expected = 28;
		result = n_bodies.get_n_total_playing();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		n_bodies.update();
		expected = 21;
		result = n_bodies.get_n_total_playing();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_total_active()
	{
		char test_func[] = "get_n_total_active";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 28;
		int result = n_bodies.get_n_total_active();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		for (int i = 0; i < BODY_TYPE_N; i++)
		{
			n_bodies.inactive[i]++;
		}
		expected = 21;
		result = n_bodies.get_n_total_active();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_total_inactive()
	{
		char test_func[] = "get_n_total_inactive";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 0;
		int result = n_bodies.get_n_total_inactive();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		for (int i = 0; i < BODY_TYPE_N; i++)
		{
			n_bodies.inactive[i]++;
		}
		expected = 7;
		result = n_bodies.get_n_total_inactive();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_total_removed()
	{
		char test_func[] = "get_n_total_removed";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 0;
		int result = n_bodies.get_n_total_removed();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		for (int i = 0; i < BODY_TYPE_N; i++)
		{
			n_bodies.removed[i]++;
		}
		expected = 7;
		result = n_bodies.get_n_total_removed();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test update()
	{
		char test_func[] = "update";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		for (int i = 0; i < BODY_TYPE_N; i++)
		{
			n_bodies.inactive[i]++;
		}
		n_bodies.update();

		int expected = 21;
		int result = n_bodies.get_n_total_playing();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		expected = 21;
		result = n_bodies.get_n_total_active();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		expected = 0;
		result = n_bodies.get_n_total_inactive();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}

		expected = 7;
		result = n_bodies.get_n_total_removed();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_massive()
	{
		char test_func[] = "get_n_massive";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 21;
		int result = n_bodies.get_n_massive();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_SI
	{
		char test_func[] = "get_n_SI";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 1+2+3+4;
		int result = n_bodies.get_n_SI();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_NSI
	{
		char test_func[] = "get_n_NSI";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 5+6;
		int result = n_bodies.get_n_NSI();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_NI
	{
		char test_func[] = "get_n_NI";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 7;
		int result = n_bodies.get_n_NI();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_GD
	{
		char test_func[] = "get_n_GD";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 5+6;
		int result = n_bodies.get_n_GD();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_MT2
	{
		char test_func[] = "get_n_MT2";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 2;
		int result = n_bodies.get_n_MT2();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_n_MT1
	{
		char test_func[] = "get_n_MT1";

		n_objects_t n_bodies(1, 2, 3, 4, 5, 6, 7);
		int expected = 3+4;
		int result = n_bodies.get_n_MT1();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected != result)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n\t\tExpected: %4d,\n\t\t But was: %4d\n", __LINE__, expected, result);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_bound_SI()
	{
		char test_func[] = "get_bound_SI";

		n_objects_t n_bodies(1, 1, 1, 0, 2, 3, 2);

		interaction_bound expected(0, 3, 0, 8);
		interaction_bound result = n_bodies.get_bound_SI();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected.sink.x   != result.sink.x   || expected.sink.y   != result.sink.y ||
			expected.source.x != result.source.x || expected.source.y != result.source.y)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n", __LINE__);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_bound_NSI()
	{
		char test_func[] = "get_bound_NSI";

		n_objects_t n_bodies(1, 1, 1, 0, 2, 3, 2);

		interaction_bound expected(3, 8, 0, 3);
		interaction_bound result = n_bodies.get_bound_NSI();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected.sink.x != result.sink.x     || expected.sink.y != result.sink.y ||
			expected.source.x != result.source.x || expected.source.y != result.source.y)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n", __LINE__);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}

	// Test get_bound_NI()
	{
		char test_func[] = "get_bound_NI";

		n_objects_t n_bodies(1, 1, 1, 0, 2, 3, 2);

		interaction_bound expected(8, 10, 0, 8);
		interaction_bound result = n_bodies.get_bound_NI();

		fprintf(stderr, "%40s(): ", test_func);
		if (expected.sink.x != result.sink.x     || expected.sink.y != result.sink.y ||
			expected.source.x != result.source.x || expected.source.y != result.source.y)
		{
			fprintf(stderr, "FAILED (Line: %4d)\n", __LINE__);
		}
		else
		{
			fprintf(stderr, "PASSED\n");
		}
	}
}
