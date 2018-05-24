#include "darknetcxx/detection.hpp"
#include "darknetcxx/box.hpp"
#include <cstring>
#include <iostream>
#include <map>
// #include <sys/time.h>

namespace darknet {

Detection::Detection(int batch,
                     int inputs,
                     int n,
                     int side,
                     int classes,
                     int coords,
                     bool rescore,
                     bool softmax,
                     bool sqrt,
                     float jitter,
                     float object_scale,
                     float noobject_scale,
                     float class_scale,
                     float coord_scale)
  : //
  m_batch(batch)
  , m_input_size(inputs)
  , m_output_size(m_input_size)
  , m_n(n)
  , m_side(side)
  , m_w(side)
  , m_h(side)
  , m_classes(classes)
  , m_coords(coords)
  ,
  // m_truth_size(side * side * (1 + coords + classes)),
  m_rescore(rescore)
  ,
  // hpyer
  // m_softmax(softmax),
  m_sqrt(sqrt)
  ,
  // m_jitter(jitter),
  m_object_scale(object_scale)
  , m_noobject_scale(noobject_scale)
  , m_class_scale(class_scale)
  , m_coord_scale(coord_scale)
  ,
  // resources
  m_cost(0)
  , m_output(batch * m_output_size)
  , m_delta(batch * m_output_size)
#ifdef DARKNET_GPU
  , m_output_gpu(batch * m_output_size)
  , m_delta_gpu(batch * m_output_size)
#endif
{
    std::cout << "Detection Layer" << std::endl;
    srand(0);
}

Detection::~Detection() = default;

void
Detection::forward(State& state)
{
    int locations = m_side * m_side;
    // TODO(mwx36mwx) use move semantics
    memcpy(m_output.get(), state.input, m_output_size * m_batch * sizeof(float));

    if (state.train) {
        float avg_anyobj = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_iou = 0;
        int count = 0;

        m_cost = 0;
        // int size = m_input_size * m_batch;
        m_delta.fill(0);

        for (int b = 0; b < m_batch; ++b) {
            int index = b * m_input_size; // start index of a batch
            // locations is 7*7
            for (int loc = 0; loc < locations; ++loc) {
                // cumpute the loss of confidence
                for (int j = 0; j < m_n; ++j) {
                    // p_index denotes the coords of predictions
                    // every grid predict m_n boxes ('num' in cfg)
                    // m_n is 3 here but 2 in paper
                    int p_index = index + locations * m_classes + loc * m_n + j;
                    // printf("%d %d %d %d %d\n", m_input_size, b, loc, j, p_index);
                    m_delta[p_index] = m_noobject_scale * (0 - m_output[p_index]);
                    // Assuming that B boxes don'y have objects at all.
                    m_cost += m_noobject_scale * pow(m_output[p_index], 2);
                    avg_anyobj += m_output[p_index];
                } // for box

                // truth_index denotes the index of truth coords
                // 1 means amount of confidential
                int truth_index = (b * locations + loc) * (1 + m_coords + m_classes);
                // TODO(mwx36mwx) how is truth ranged
                int is_obj = state.truth[truth_index];
                if (!is_obj)
                    continue;

                /***********************************************************************
                 * class delta
                 **********************************************************************/

                int class_index = index + loc * m_classes;
                for (int cla = 0; cla < m_classes; ++cla) {
                    int truth_class_index = truth_index + 1 + cla;
                    int output_class_index = class_index + cla;

                    float cla_delta = state.truth[truth_class_index] - m_output[output_class_index];

                    m_delta[output_class_index] = m_class_scale * cla_delta;
                    m_cost += m_class_scale * pow(cla_delta, 2);

                    // TODO(mwx36mwx) FIXME FIXME !? float to bool ?????????
                    if (state.truth[truth_index + 1 + cla]) {
                        avg_cat += m_output[output_class_index];
                    }
                    avg_allcat += m_output[output_class_index];
                } // for class

                /***********************************************************************
                 * box iou delta
                 **********************************************************************/

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                box truth(state.truth + truth_index + 1 + m_classes); // xywh
                truth.x /= m_side;
                truth.y /= m_side;

                // find final pretict box since the paper says
                // We only predict one set of class probabilities per grid cell,
                // regardless of the number of boxes B
                for (int ibox = 0; ibox < m_n; ++ibox) {
                    int box_index = index + locations * (m_classes + m_n) + (loc * m_n + ibox) * m_coords;

                    box out(m_output.get() + box_index);
                    out.x /= m_side;
                    out.y /= m_side;

                    if (m_sqrt) {
                        out.w *= out.w;
                        out.h *= out.h;
                    }

                    // compute IoU
                    float iou = box::iou(out, truth);
                    // compute RMSE (Root Mean-Square Error)
                    float rmse = box::rmse(out, truth);

                    // find the box who has max IoU or min RMSE
                    if (best_iou > 0 || iou > 0) {
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_index = ibox;
                        }
                    } else {
                        if (rmse < best_rmse) {
                            best_rmse = rmse;
                            best_index = ibox;
                        }
                    }
                } // for box

                // index for predicted box
                int box_index = index + locations * (m_classes + m_n) + (loc * m_n + best_index) * m_coords;
                // index for truth box
                int tbox_index = truth_index + 1 + m_classes;

                box out(m_output.get() + box_index);
                out.x /= m_side;
                out.y /= m_side;

                if (m_sqrt) {
                    out.w *= out.w;
                    out.h *= out.h;
                }
                float iou = box::iou(out, truth);

                // Minus the box who has object
                int p_index = index + locations * m_classes + loc * m_n + best_index;
                m_cost -= m_noobject_scale * pow(m_output[p_index], 2);
                m_cost += m_object_scale * pow(1 - m_output[p_index], 2);
                avg_obj += m_output[p_index];
                m_delta[p_index] = m_object_scale * (1. - m_output[p_index]);

                if (m_rescore) {
                    m_delta[p_index] = m_object_scale * (iou - m_output[p_index]);
                }

                m_delta[box_index + 0] = m_coord_scale * (state.truth[tbox_index + 0] - m_output[box_index + 0]);
                m_delta[box_index + 1] = m_coord_scale * (state.truth[tbox_index + 1] - m_output[box_index + 1]);
                m_delta[box_index + 2] = m_coord_scale * (state.truth[tbox_index + 2] - m_output[box_index + 2]);
                m_delta[box_index + 3] = m_coord_scale * (state.truth[tbox_index + 3] - m_output[box_index + 3]);
                if (m_sqrt) {
                    m_delta[box_index + 2] = m_coord_scale * (sqrt(state.truth[tbox_index + 2]) - m_output[box_index + 2]);
                    m_delta[box_index + 3] = m_coord_scale * (sqrt(state.truth[tbox_index + 3]) - m_output[box_index + 3]);
                }

                // don't use
                m_cost += pow(1 - iou, 2);
                avg_iou += iou;
                ++count;
            } // for locations
        }     // for batch

        m_cost = pow(mag_array(m_delta.get(), m_output_size * m_batch), 2);

        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any "
               "Obj: %f, count: %d\n",
               avg_iou / count,
               avg_cat / count,
               avg_allcat / (count * m_classes),
               avg_obj / count,
               avg_anyobj / (m_batch * locations * m_n),
               count);
    } // if (state.train)
}

void
Detection::backward(State& state)
{
    axpy_cpu(m_batch * m_input_size, 1.f, m_delta.get(), state.delta);
}

#ifdef DARKNET_GPU
void
Detection::forward_gpu(State& state)
{

    // if (!state.train) {
    //     copy_ongpu(m_batch * m_input_size, state.input, m_output_gpu.get());
    // }
    if (!state.train) {
        // struct timeval t0, t1;
        // long elapsed;
        // gettimeofday(&t0, 0);
        cuda_pull_array(state.input, m_output.get(), m_batch * m_output_size);
        // gettimeofday(&t1, 0);
        // elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
        // std::cout << "cuda_pull_array time: " << elapsed / 1000.0 << " ms" << std::endl;
        return;
    } else {
        float* in_cpu = new float[m_batch * m_input_size];
        float* truth_cpu = 0;
        if (state.truth) {
            int num_truth = m_batch * m_side * m_side * (1 + m_coords + m_classes);
            truth_cpu = new float[num_truth];
            cuda_pull_array(state.truth, truth_cpu, num_truth);
        }
        cuda_pull_array(state.input, in_cpu, m_batch * m_input_size);
        State cpu_state;
        cpu_state.train = state.train;
        cpu_state.truth = truth_cpu;
        cpu_state.input = in_cpu;

        forward(cpu_state);
        cuda_push_array(m_output_gpu.get(), m_output.get(), m_batch * m_output_size);
        cuda_push_array(m_delta_gpu.get(), m_delta.get(), m_batch * m_input_size);

        delete[] cpu_state.input;
        if (cpu_state.truth)
            delete[] cpu_state.truth;
    }
}

void
Detection::backward_gpu(State& state)
{
    axpy_ongpu(m_batch * m_input_size, 1.f, m_delta_gpu.get(), state.delta_gpu);
}
#endif

void
Detection::get_detection_boxes(int w, int h, float thresh, float** probs, box* boxes, int only_objectness)
{
    float* predictions = m_output.get();

    for (int i = 0; i < m_side * m_side; ++i) {
        int row = i / m_side;
        int col = i % m_side;
        for (int n = 0; n < m_n; ++n) {
            int index = i * m_n + n;
            int p_index = m_side * m_side * m_classes + i * m_n + n;
            float scale = predictions[p_index];
            int box_index = m_side * m_side * (m_classes + m_n) + (i * m_n + n) * 4;
            boxes[index].x = (predictions[box_index + 0] + col) / m_side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / m_side * h;
            boxes[index].w = pow(predictions[box_index + 2], (m_sqrt ? 2 : 1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (m_sqrt ? 2 : 1)) * h;

            for (int j = 0; j < m_classes; ++j) {
                int class_index = i * m_classes;
                float prob = scale * predictions[class_index + j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if (only_objectness) {
                probs[index][0] = scale;
            }
        } // for box
    }     // for loc
}

/*******************************************************************************
 * Getters
 ******************************************************************************/

Ptr<float>&
Detection::output()
{
    return m_output;
}
Ptr<float>&
Detection::delta()
{
    return m_delta;
}

#ifdef DARKNET_GPU
Ptr_gpu<float>&
Detection::output_gpu()
{
    return m_output_gpu;
}
Ptr_gpu<float>&
Detection::delta_gpu()
{
    return m_delta_gpu;
}
#endif

std::map<std::string, int>
Detection::get_params()
{
    std::map<std::string, int> params = { { "out_h", m_h }, { "out_w", m_w }, { "out_c", m_n }, { "side", m_side }, { "n", m_n }, { "classes", m_classes } };
    return params;
}

LayerType
Detection::type()
{
    return LayerType::DETECTION;
}

/**********
 * Setters *
 **********/

void
Detection::set_batch(const int batch)
{
    m_batch = batch;
    m_output.resize(batch * m_output_size);
    m_delta.resize(batch * m_output_size);
#ifdef DARKNET_GPU
    m_output_gpu.resize(batch * m_output_size);
    m_delta_gpu.resize(batch * m_output_size);
#endif
}

} // namespace darknet
