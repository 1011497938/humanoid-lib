#include "darknetcxx/box.hpp"
#include <cmath>
#include <cstdlib>

namespace darknet {
dbox::dbox()
  : dx(0)
  , dy(0)
  , dw(0)
  , dh(0)
{
}

dbox::dbox(int dx_, int dy_, int dw_, int dh_)
  : dx(dx_)
  , dy(dy_)
  , dw(dw_)
  , dh(dh_)
{
}

box::box(int x_, int y_, int w_, int h_)
  : x(x_)
  , y(y_)
  , w(w_)
  , h(h_)
{
}

box::box(float* f)
  : x(f[0])
  , y(f[1])
  , w(f[2])
  , h(f[3])
{
}

/*******************************************************************************
 * Public interface
 ******************************************************************************/
float
box::iou(box& a, box& b)
{
    return intersection(a, b) / union_(a, b);
}

float
box::rmse(box& a, box& b)
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.w - b.w, 2) + pow(a.h - b.h, 2));
}

dbox
box::diou(box& a, box& b)
{
    float u = union_(a, b);
    float i = intersection(a, b);
    dbox di = dintersect(a, b);
    dbox du = dunion(a, b);
    dbox dd;

    if (i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2 * pow((1 - (i / u)), 1) * (di.dx * u - du.dx * i) / (u * u);
    dd.dy = 2 * pow((1 - (i / u)), 1) * (di.dy * u - du.dy * i) / (u * u);
    dd.dw = 2 * pow((1 - (i / u)), 1) * (di.dw * u - du.dw * i) / (u * u);
    dd.dh = 2 * pow((1 - (i / u)), 1) * (di.dh * u - du.dh * i) / (u * u);
    return dd;
}

typedef struct
{
    int index;
    int class_;
    float** probs;
} sortable_bbox;

int
nms_comparator(const void* pa, const void* pb)
{
    sortable_bbox a = *(sortable_bbox*)pa;
    sortable_bbox b = *(sortable_bbox*)pb;
    float diff = a.probs[a.index][b.class_] - b.probs[b.index][b.class_];
    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

void
box::nms_sort(box* boxes, float** probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox* s = new sortable_bbox[total];

    for (i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].class_ = 0;
        s[i].probs = probs;
    }

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            s[i].class_ = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for (i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0)
                continue;
            box a = boxes[s[i].index];
            for (j = i + 1; j < total; ++j) {
                box b = boxes[s[j].index];
                if (iou(a, b) > thresh) {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    delete[] s;
}

void
box::do_nms(box* boxes, float** probs, int total, int classes, float thresh)
{
    int i, j, k;
    for (i = 0; i < total; ++i) {
        int any = 0;
        for (k = 0; k < classes; ++k)
            any = any || (probs[i][k] > 0);
        if (!any) {
            continue;
        }
        for (j = i + 1; j < total; ++j) {
            if (iou(boxes[i], boxes[j]) > thresh) {
                for (k = 0; k < classes; ++k) {
                    if (probs[i][k] < probs[j][k])
                        probs[i][k] = 0;
                    else
                        probs[j][k] = 0;
                }
            }
        }
    }
}

/*******************************************************************************
 * Private functions
 ******************************************************************************/
float
box::overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float
box::intersection(box& a, box& b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0;

    float area_ = w * h;
    return area_;
}

float
box::union_(box& a, box& b)
{
    return a.area() + b.area() - intersection(a, b);
}

dbox
box::derivative(box& a, box& b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w / 2;
    float l2 = b.x - b.w / 2;
    if (l1 > l2) {
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w / 2;
    float r2 = b.x + b.w / 2;
    if (r1 < r2) {
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2) {
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h / 2;
    float t2 = b.y - b.h / 2;
    if (t1 > t2) {
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h / 2;
    float b2 = b.y + b.h / 2;
    if (b1 < b2) {
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2) {
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}

dbox
box::dintersect(box& a, box& b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw * h;
    di.dx = dover.dx * h;
    di.dh = dover.dh * w;
    di.dy = dover.dy * w;

    return di;
}

dbox
box::dunion(box& a, box& b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}

bbox::bbox() {
}

bbox::bbox(int label, float prob, int left, int top, int right, int bottom)
  : m_label(label)
  , m_prob(prob)
  , m_left(left)
  , m_top(top)
  , m_right(right)
  , m_bottom(bottom)
{
}

RelativeBBox::RelativeBBox(int label, float prob, float x, float y, float h, float w)
  : m_label(label)
  , m_prob(prob)
  , m_x(x)
  , m_y(y)
  , m_h(h)
  , m_w(w)
{
}

} // namespace darknet
