#pragma once
#include <vector>
namespace darknet {

struct dbox {
  dbox();
  explicit dbox(int dx, int dy, int dw, int dh);

  float dx, dy, dw, dh;
};

struct box {
  explicit box(int x, int y, int w, int h);
  explicit box(float *f);

  static float iou(box &a, box &b);
  static float rmse(box &a, box &b); // root mean square error
  static dbox diou(box &a, box &b);

  void nms_sort(box *boxes, float **probs, int total, int classes,
                float thresh);
  void do_nms(box *boxes, float **probs, int total, int classes, float thresh);

  inline float area() { return w * h; }

  static dbox derivative(box &a, box &b);
  static float overlap(float x1, float w1, float x2, float w2);
  static float intersection(box &a, box &b);
  static float union_(box &a, box &b);
  static dbox dintersect(box &a, box &b);
  static dbox dunion(box &a, box &b);

  float x, y, w, h;
};

struct bbox {
  bbox();
  bbox(int label, float prob, int left, int top, int right, int bottom);
  int m_label;
  float m_prob;
  int m_left, m_top, m_right, m_bottom;
};

struct RelativeBBox {
  RelativeBBox(int label, float prob, float x, float y, float h, float w);
  int m_label;
  float m_prob;
  float m_x, m_y, m_h, m_w;
};

} // namespace darknet
