// /**
//  * © Copyright IBM Corporation 2024. All Rights Reserved.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *      http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <ios>
#include <iostream>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "main-helpers.h"

#define NY_DISTANCE 8

std::vector<std::vector<float>> load_vectors_from_csv_safe(size_t &vector_dim)
{
    std::ifstream file("dataset/train.csv");
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file: train.csv" << std::endl;
        exit(1);
    }

    std::vector<std::vector<float>> data;
    std::string line;
    size_t line_number = 0;
    vector_dim = 0;

    while (std::getline(file, line))
    {
        line_number++;
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos)
        {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row_data;

        while (std::getline(ss, cell, ','))
        {
            try
            {
                row_data.push_back(std::stof(cell));
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Error: Invalid number format on line " << line_number << ": \"" << cell << "\"" << std::endl;
                exit(1);
            }
            catch (const std::out_of_range &e)
            {
                std::cerr << "Error: Number out of range on line " << line_number << ": \"" << cell << "\"" << std::endl;
                exit(1);
            }
        }

        if (data.empty())
        {
            if (row_data.empty())
                continue;
            vector_dim = row_data.size();
        }
        else if (row_data.size() != vector_dim)
        {
            std::cerr << "Error: Inconsistent row size at line " << line_number
                      << ". Expected " << vector_dim << " but got " << row_data.size() << "." << std::endl;
            exit(1);
        }

        data.push_back(std::move(row_data));
    }

    return data;
}

int main(int argc, char *argv[])
{
    int rtn;

    int *array_sizes;
    int size, num_array_sizes = 0;
    struct results_data_t *results;
    char group_id_name[GROUP_ID_MAX][GROUP_ID_NAME_MAX];

    long long int array_index;
    float **x_d = (float **)malloc(sizeof(float *));
    float **y0_d = (float **)malloc(sizeof(float *));
    float **y1_d = (float **)malloc(sizeof(float *));
    float **y2_d = (float **)malloc(sizeof(float *));
    float **y3_d = (float **)malloc(sizeof(float *));
    int8_t **xi_d = (int8_t **)malloc(sizeof(int8_t *));
    int8_t **yi_d = (int8_t **)malloc(sizeof(int8_t *));
    uint8_t **c1_d = (uint8_t **)malloc(sizeof(uint8_t *));
    uint8_t **c2_d = (uint8_t **)malloc(sizeof(uint8_t *));

    float dp0 = 0.0f;
    float dp1 = 0.0f;
    float dp2 = 0.0f;
    float dp3 = 0.0f;
    struct flags_t cmd_flags;

    rtn = read_cmd_opts(argc, argv, &cmd_flags);
    if (rtn)
    {
        std::cout << "ERROR reading command line args\n";
    }
    std::filesystem::create_directories("./results");
    std::string dateSuffix = "_" + getDateAsFileSuffix() + ".txt";

    std::string RESULTS_OUTPUT = "results/test_results" + dateSuffix;
    std::string TIME_OUTPUT = "results/test_time" + dateSuffix;

    array_sizes = cmd_flags.array_sizes;
    num_array_sizes = cmd_flags.num_array_sizes;
    check_array_index(num_array_sizes);

    results = (struct results_data_t *)malloc(sizeof(results_data_t) * FUNC_ID_MAX);
    if (!results)
    {
        std::cerr << "Failed to allocate memory for results.\n";
        return -1;
    }

    initialize_group_func_names(group_id_name, results);

    if (cmd_flags.verbose_output)
    {
        print_cmd_opts(cmd_flags, results, group_id_name);
    }
    if (cmd_flags.run_custom)
    {
        std::cout << "Running custom test..." << std::endl;

        size_t vector_dim;
        std::vector<std::vector<float>> custom_data = load_vectors_from_csv_safe(vector_dim);
        size_t num_vectors = custom_data.size();
        size_t array_size = vector_dim;

        std::cout << num_vectors << " vectors loaded with dimension " << array_size << std::endl;

        std::string custom_results_filename = "results/custom_results" + dateSuffix;
        std::string mismatch_results_filename = "results/custom_mismatches" + dateSuffix;
        std::string scalar_products_filename = "results/scalar_products" + dateSuffix;
        std::string vector_products_filename = "results/vector_products" + dateSuffix;

        std::ofstream custom_results_file(custom_results_filename);
        std::ofstream mismatch_results_file(mismatch_results_filename);
        std::ofstream scalar_products_file(scalar_products_filename);
        std::ofstream vector_products_file(vector_products_filename);

        if (!custom_results_file || !mismatch_results_file || !scalar_products_file || !vector_products_file)
        {
            std::cerr << "Error: Could not open one of the result files. "
                      << strerror(errno) << std::endl;
            exit(-1);
        }

        if (!custom_results_file || !mismatch_results_file || !scalar_products_file || !vector_products_file)
        {
            std::cerr << "Error: Could not open one of the result files. "
                      << strerror(errno) << std::endl;
            exit(-1);
        }

        auto write_headers = [&]()
        {
            custom_results_file << "index\tscalar_hex\tvector_hex\tscalar_val\tvector_val\tsv_iulps\n";
            mismatch_results_file << "index\tscalar_hex\tvector_hex\tscalar_val\tvector_val\tsv_iulps\tx_vector\ty_vector\n";
            scalar_products_file << "index\tscalar_hex\tscalar_val\n";
            vector_products_file << "index\tvector_hex\tvector_val\n";
        };
        write_headers();

        float max_scalar = -std::numeric_limits<float>::infinity();
        float min_scalar = std::numeric_limits<float>::infinity();
        float max_vector = -std::numeric_limits<float>::infinity();
        float min_vector = std::numeric_limits<float>::infinity();
        float max_diff = 0.0f;
        size_t max_diff_index = 0;

        for (size_t i = 0; i + 1 < num_vectors; i += 2)
        {
            const float *x = custom_data[i].data();
            const float *y = custom_data[i + 1].data();

            float scalar = 0.0f;
            float vector = 0.0f;

            if (cmd_flags.run_func_flag[FVEC_L2SQR_REF])
            {
                scalar = base::fvec_L2sqr_ref(x, y, array_size);
                vector = powerpc::fvec_L2sqr_ref_ppc(x, y, array_size);
            }
            else if (cmd_flags.run_func_flag[FVEC_INNER_PRODUCT_REF])
            {
                scalar = base::fvec_inner_product_ref(x, y, array_size);
                vector = powerpc::fvec_inner_product_ref_ppc(x, y, array_size);
            }
            else if (cmd_flags.run_func_flag[FVEC_L1_REF])
            {
                scalar = base::fvec_L1_ref(x, y, array_size);
                vector = powerpc::fvec_L1_ref_ppc(x, y, array_size);
            }
            else if (cmd_flags.run_func_flag[COSINE_DISTANCE_REF])
            {
                scalar = base::cosine_distance_ref(x, y, array_size);
                vector = powerpc::cosine_distance_ref_ppc(x, y, array_size);
            }
            else if (cmd_flags.run_func_flag[JACCARD_DISTANCE_REF])
            {
                scalar = base::jaccard_distance_ref(x, y, array_size);
                vector = powerpc::jaccard_distance_ref_ppc(x, y, array_size);
            }
            else
            {
                std::cerr << "Error: No valid distance function selected for custom test." << std::endl;
                return -1;
            }

            union
            {
                float f;
                int32_t i;
            } u, v;
            u.f = scalar;
            v.f = vector;
            int sv_iulps = u.i - v.i;

            max_scalar = std::max(max_scalar, scalar);
            min_scalar = std::min(min_scalar, scalar);
            max_vector = std::max(max_vector, vector);
            min_vector = std::min(min_vector, vector);

            float abs_diff = std::fabs(scalar - vector);
            if (abs_diff > max_diff)
            {
                max_diff = abs_diff;
                max_diff_index = i;
            }

            // Set stream formats for output
            auto &cr_file = custom_results_file;
            cr_file << i << "\t"
                    << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << u.i << "\t"
                    << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << v.i << "\t"
                    << std::dec << std::scientific << std::setprecision(10) << scalar << "\t"
                    << std::scientific << std::setprecision(10) << vector << "\t"
                    << std::dec << sv_iulps << "\n";

            scalar_products_file << i << "\t" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << u.i << "\t" << std::dec << std::scientific << std::setprecision(10) << scalar << "\n";
            vector_products_file << i << "\t" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << v.i << "\t" << std::dec << std::scientific << std::setprecision(10) << vector << "\n";

            if (sv_iulps != 0)
            {
                std::ostringstream x_stream, y_stream;
                for (size_t d = 0; d < array_size; ++d)
                {
                    x_stream << x[d] << (d < array_size - 1 ? " " : "");
                    y_stream << y[d] << (d < array_size - 1 ? " " : "");
                }

                auto &mm_file = mismatch_results_file;
                mm_file << i << "\t"
                        << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << u.i << "\t"
                        << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << v.i << "\t"
                        << std::dec << std::scientific << std::setprecision(10) << scalar << "\t"
                        << std::scientific << std::setprecision(10) << vector << "\t"
                        << std::dec << sv_iulps << "\t"
                        << x_stream.str() << "\t"
                        << y_stream.str() << "\n";
            }
        }

        std::cout << "Custom test completed." << std::endl;
        std::cout << "Max Scalar Value: " << max_scalar << std::endl;
        std::cout << "Min Scalar Value: " << min_scalar << std::endl;
        std::cout << "Max Vector Value: " << max_vector << std::endl;
        std::cout << "Min Vector Value: " << min_vector << std::endl;
        std::cout << "Max Absolute Difference: " << max_diff << " at index " << max_diff_index << std::endl;
    }
    else
    {
        std::ofstream timefile(TIME_OUTPUT);
        std::ofstream resultfile(RESULTS_OUTPUT);
        if (!timefile)
        {
            std::cout << "Could not open output file " << TIME_OUTPUT << " exiting.\n";
            exit(-1);
        }
        if (!resultfile)
        {
            std::cout << "Could not open output file " << RESULTS_OUTPUT
                      << " exiting.\n";
            exit(-1);
        }
        for (array_index = 0; array_index < num_array_sizes; array_index++)
        {
            /* Test the various distance functions in euclidean_l2_distance.cc  */
            /* Setup input arrays for the various euclidian distance tests */

            size = array_sizes[array_index];

            std::cout << "Running array size " << size << std::endl;

            load_data_float(size, x_d, y0_d, y1_d, y2_d, y3_d);

            const float *x = *x_d;
            const float *y0 = *y0_d;
            const float *y1 = *y1_d;
            const float *y2 = *y2_d;
            const float *y3 = *y3_d;

            load_data_int8(size, xi_d, yi_d);

            const int8_t *xi = *xi_d;
            const int8_t *yi = *yi_d;

            load_data_char(size, c1_d, c2_d);

            const uint8_t *c1 = *c1_d;
            const uint8_t *c2 = *c2_d;

            float *dis = (float *)malloc(sizeof(float *) * NY_DISTANCE);

            /**********  Eulcidian tests *************/

            /* Test fvec_L2sqr_ref  */
            if (cmd_flags.run_func_flag[FVEC_L2SQR_REF])
                test_fvec_L2sqr_ref(results, FVEC_L2SQR_REF, array_index,
                                    cmd_flags.num_runs,
                                    cmd_flags.run_code_version, x, y2, size);

            /* Test fvec_norm_L2sqr_ref  */
            if (cmd_flags.run_func_flag[FVEC_NORM_L2SQR_REF])
                test_fvec_norm_L2sqr_ref(results, FVEC_NORM_L2SQR_REF,
                                         array_index, cmd_flags.num_runs,
                                         cmd_flags.run_code_version, x,
                                         size);

            /* Test fvec_L2sqr_ny_transposed_ref  */
            if (cmd_flags.run_func_flag[FVEC_L2SQR_NY_TRANSPOSED_REF])
                test_fvec_L2sqr_ny_transposed_ref(results,
                                                  FVEC_L2SQR_NY_TRANSPOSED_REF,
                                                  array_index, cmd_flags.num_runs,
                                                  cmd_flags.run_code_version,
                                                  dis, x, y1, y2,
                                                  (size_t)(size / 4), 2,
                                                  NY_DISTANCE);

            /* Test fvec_L2sqr_batch_4_ref   */
            if (cmd_flags.run_func_flag[FVEC_L2SQR_BATCH_4_REF])
                test_fvec_L2sqr_batch_4_ref(results, FVEC_L2SQR_BATCH_4_REF,
                                            array_index, cmd_flags.num_runs,
                                            cmd_flags.run_code_version, x, y0, y1,
                                            y2, y3, size, dp0, dp1, dp2, dp3);

            /* Test ivec_L2sqr_ref  */
            if (cmd_flags.run_func_flag[IVEC_L2SQR_REF])
                test_ivec_L2sqr_ref(results, IVEC_L2SQR_REF, array_index,
                                    cmd_flags.num_runs,
                                    cmd_flags.run_code_version, xi, yi, size);

            /**********  Inner product tests *************/
            /* Test inner_product_ref  */
            if (cmd_flags.run_func_flag[FVEC_INNER_PRODUCT_REF])
                test_fvec_inner_product_ref(results, FVEC_INNER_PRODUCT_REF,
                                            array_index, cmd_flags.num_runs,
                                            cmd_flags.run_code_version, x, y2,
                                            size);

            /* Test ivec_inner_product_batch_4_ref  */
            if (cmd_flags.run_func_flag[FVEC_INNER_PRODUCT_BATCH_4_REF])
                test_fvec_inner_product_batch_4_ref(results,
                                                    FVEC_INNER_PRODUCT_BATCH_4_REF,
                                                    array_index,
                                                    cmd_flags.num_runs,
                                                    cmd_flags.run_code_version,
                                                    x, y0, y1, y2, y3, size,
                                                    dp0, dp1, dp2, dp3);

            /* Test ivec_inner_product_ref  */
            if (cmd_flags.run_func_flag[IVEC_INNER_PRODUCT_REF])
                test_ivec_inner_product_ref(results, IVEC_INNER_PRODUCT_REF, array_index,
                                            cmd_flags.num_runs,
                                            cmd_flags.run_code_version, xi, yi, size);

            /**********  Manhattan distance tests *************/

            if (cmd_flags.run_func_flag[FVEC_L1_REF])
            {
                test_fvec_L1_ref(results, FVEC_L1_REF, array_index,
                                 cmd_flags.num_runs,
                                 cmd_flags.run_code_version, x, y0, size);
            }

            /**********  Cosine distance test *************/

            if (cmd_flags.run_func_flag[COSINE_DISTANCE_REF])
            {
                test_cosine_distance_ref(results, COSINE_DISTANCE_REF, array_index,
                                         cmd_flags.num_runs,
                                         cmd_flags.run_code_version, x, y0, size);
            }

            /**********  Hamming distance test *************/

            if (cmd_flags.run_func_flag[HAMMING_DISTANCE_REF])
            {
                test_hamming_distance_ref(results, HAMMING_DISTANCE_REF,
                                          array_index, cmd_flags.num_runs,
                                          cmd_flags.run_code_version, c1, c2,
                                          size);
            }

            /**********  Jaccard distance test *************/

            if (cmd_flags.run_func_flag[JACCARD_DISTANCE_REF])
            {
                test_jaccard_distance_ref(results, JACCARD_DISTANCE_REF,
                                          array_index,
                                          cmd_flags.num_runs,
                                          cmd_flags.run_code_version, x, y0,
                                          size);
            }

            /* Release data arrays.  */
            release_data_float(x_d, y0_d, y1_d, y2_d, y3_d, dis);
            release_data_int8(xi_d, yi_d);
            release_data_char(c1_d, c2_d);
        }

        /* Print results */
        print_time(timefile, FUNC_ID_MAX, array_index, results, cmd_flags,
                   group_id_name);
        print_result(resultfile, FUNC_ID_MAX, array_index, results, cmd_flags,
                     group_id_name);

        /* Release results array.  */
        free(results);

        timefile.close();
        resultfile.close();

        return 0;
    }
}