/**
* Implementation of KalmanFilter class.
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/

#include <iostream>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    double dt,
    const Eigen::MatrixXd& A, // state-transition model
    const Eigen::MatrixXd& C, // observation model
    const Eigen::MatrixXd& Q, // covariance process noise
    const Eigen::MatrixXd& R, // covariance observation noise
    const Eigen::MatrixXd& P) // initial estimate covariance
  : A(A), C(C), Q0(Q), R0(R), P0(P),
    m(C.rows()), n(A.rows()), dt(dt), initialized(false),
    I(n, n), x_hat(n), x_hat_new(n), mres_post(m), mres_pre(m)
{
  I.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
  x_hat = x0;
  P = P0;
  R = R0;
  Q = Q0;
  mres_pre_mag = 0.0;
  mres_post_mag = 0.0;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void KalmanFilter::init() {
  x_hat.setZero();
  P = P0;
  R = R0;
  Q = Q0;
  mres_pre_mag = 0.0;
  mres_post_mag = 0.0;
  t0 = 0;
  t = t0;
  initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd& y) {

  if(!initialized)
    throw std::runtime_error("Filter is not initialized!");

  x_hat_new = A * x_hat;
  P = A*P*A.transpose() + Q; // predicted state estimate
  S = C*P*C.transpose() + R; // innovation covariance
  K = P*C.transpose()*S.inverse(); // Kalman gain
  mres_pre = y - C*x_hat_new; // measurement pre-fit residual
  mres_pre_mag = mres_pre.norm();
  x_hat_new += K * mres_pre; // updated state estimate
  P = (I - K*C)*P; // updated estimate covariance
  x_hat = x_hat_new;

  mres_post = y - C * x_hat_new; // measurement post-fit residual
  mres_post_mag = mres_post.norm(); // measurement residual magnitude

  // Q adaption
  double factor = 0.3; // forgetting factor
  if(mres_pre_mag > 0.5) {
    Q = Q0;
  } else if(mres_pre_mag < 0.2) {
    Q = factor*(Q - (K*mres_pre) * (K*mres_pre).transpose()) + (1-factor)*Q0;
  } else {
    Q = factor*(Q + (K*mres_pre) * (K*mres_pre).transpose()) + (1-factor)*Q0;
  }

  t += dt;
}

void KalmanFilter::update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd A) {

  this->A = A;
  this->dt = dt;
  update(y);
}
