#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  bool is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = x_.size();

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Number of sigma points
  n_sigma_ = 2 * n_aug_ + 1;

  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  // Lambda for spreading factor
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(n_sigma_);

  R_radar_ = MatrixXd(3, 3);

  R_radar_ << std_radrd_ * std_radrd_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  cerr << "Starting process measurement" << endl;
  if (!is_initialized_) {
    // Initialize the covariance matrix
    P_ << 0.15, 0,    0, 0, 0,
          0,    0.15, 0, 0, 0,
          0,    0,    1, 0, 0,
          0,    0,    0, 1, 0,
          0,    0,    0, 0, 1;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      // Convert coordinates from polar to cartesian
      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double x_dot = rho_dot * cos(phi);
      double y_dot = rho_dot * sin(phi);
      double v = sqrt(x_dot * x_dot + y_dot * y_dot);

      // Store values to vector.
      x_ << x, y, v, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;

      // In some cases raw_measurements can be very very small, we round them here
      if (fabs(x_(0)) < 1e-4 && fabs(x_(1)) < 1e-4) {
        x_(0) = 1e-4;
        x_(1) = 1e-4;
      }
    } else {
      cerr << "Unknown measurement pack received, don't know how to handle" << endl;
    }

    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < weights_.size(); ++i) {
      weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  /* 1. generate sigma points
   * 2. predict sigma points
   * 3. predict mean and covariance.
   */
  double delta_t = (meas_package.timestamp_ - time_us_) / 1e+6;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);

  cerr << "About to update" << endl;

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  } else {
    cerr << "Unknown measurement pack received, don't know how to handle" << endl;
  }
  cerr << "Done updating" << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug = MatrixXd::Zero(P_aug.rows(), P_aug.cols());
  P_aug.topLeftCorner(P_.cols(), P_.rows()) = P_;
  P_aug(P_.cols(), P_.rows()) = std_a_ * std_a_;
  P_aug(P_.cols() + 1, P_.rows() + 1) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd Psq = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  float coef_aug = sqrt(lambda_ + n_aug_);

  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + coef_aug * Psq.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - coef_aug * Psq.col(i);
  }

  for (int i = 0; i < Xsig_aug.cols(); ++i) {
    double x = Xsig_aug(0, i);
    double y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yaw_d = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double x_p, y_p;

    if (fabs(yaw_d) > 1e-3) {
      x_p = x + v / yaw_d * (sin(yaw + yaw_d * delta_t) - sin(yaw));
      y_p = y + v / yaw_d * (cos(yaw) - cos(yaw + yaw_d * delta_t));
    } else {
      x_p = x + v * cos(yaw);
      y_p = y + v * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yaw_d * delta_t;
    double yaw_d_p = yaw_d;

    // add noise
    x_p += 0.5 * nu_a* delta_t * delta_t * cos(yaw);
    y_p += 0.5 * nu_a* delta_t * delta_t * sin(yaw);
    v_p += nu_a * delta_t;

    yaw_p += 0.5 * nu_yawdd * delta_t * delta_t;
    yaw_d_p += nu_yawdd * delta_t;

    // Write predicted sigma points to right column
    Xsig_pred_(0, i) = x_p;
    Xsig_pred_(1, i) = y_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yaw_d_p;
  }

  // set weights_
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights_
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  MatrixXd Z_sigma = Xsig_pred_.block(0, 0, n_z, n_sigma_);
  UpdateUKF(meas_package, Z_sigma, n_z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Radar has three measurement dimensions
  int n_z_ = 3;
  MatrixXd Z_sigma = MatrixXd(n_z_, n_sigma_);
  // Transform sigma points into measurement
  for (int i = 0; i < n_sigma_; ++i) {
    double x = Xsig_pred_(0, i);
    double y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double x_dot = cos(yaw) * v;
    double y_dot = sin(yaw) * v;

    // Radar measurement model
    Z_sigma(0, i) = sqrt(x * x + y * y);
    Z_sigma(1, i) = atan2(x, y);
    Z_sigma(2, i) = (x * x_dot + y * y_dot) / Z_sigma(0, i);
  }

  UpdateUKF(meas_package, Z_sigma, n_z_);
}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Z_sigma, int n_z)
{
  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred = Z_sigma * weights_;

  // Measurement covariance matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sigma_; ++i) {
    VectorXd z_diff = Z_sigma.col(i) - z_pred;
    NormalizeAngle(&(z_diff(1)));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise
  MatrixXd R = MatrixXd(n_z, n_z);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    R = R_radar_;
  } else {
    R = R_lidar_;
  }

  S = S + R;

  // Calculate cross correlations
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.);

  for (int i = 0; i < n_sigma_; ++i) {
    VectorXd z_diff = Z_sigma.col(i)- z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      NormalizeAngle(&(z_diff(1)));
    }
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(&(x_diff(3)));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  // Measurement
  VectorXd z = meas_package.raw_measurements_;
  // Kalman gain
  MatrixXd K = Tc * S.inverse();
  // Residual
  VectorXd z_diff = z - z_pred;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    NormalizeAngle(&(z_diff(1)));
  }
  // Update state mean and covariance
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  // Calculate NIS
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    NIS_radar_ = z.transpose() * S.inverse() * z;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    NIS_laser_ = z.transpose() * S.inverse() * z;
  }
}

void UKF::NormalizeAngle(double *angle) {
  while (*angle > M_PI) *angle -= 2. * M_PI;
  while (*angle < -M_PI) *angle += 2. * M_PI;
}
