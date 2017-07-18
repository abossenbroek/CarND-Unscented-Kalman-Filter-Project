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
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

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
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = n_x_ + 2;

    // Lambda for spreading factor
    lambda_ = 3 - n_aug_;

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
    */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    if (!is_initialized_) {
        x_ = VectorXd(n_x_);
        x_ = VectorXd::setOnes();

        time_us_ = meas_package.timestamp_;

        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

        } else {

        }

        return;
    }

    /* 1. generate sigma points
     * 2. predict sigma points
     * 3. predict mean and covariance.
     */
    //calculate square root of P
    MatrixXd A = P_.llt().matrixL();
    //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

    VectorXd x = meas_package.raw_measurements_;
    // TODO: change for radar / lidar
    // TODO: do we really need Xsig??
    Xsig.col(0) = x;

    float coef = sqrt(lambda_ + n_x_);

    for (int i = 0; i < n_x_; ++i) {
        Xsig.col(i + 1) = x + coef * A.col(i);
        Xsig.col(i + 1 + n_x_) = x - coef * A.col(i);
    }
    // augmentation


    double delta_t = (meas_package.timestamp_ - time_us_) / 1e+6;
    Prediction(delta_t);
    time_us_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

    } else {

    }

    /**
    TODO:

    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */
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
        double vel = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yaw_d = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        VectorXd pred = VectorXd(n_x_);
        VectorXd noise = VectorXd(n_x_);

        double dt_hlf = 0.5 * delta_t * delta_t;

        noise << dt_hlf * cos(yaw) * nu_a,
                dt_hlf * sin(yaw) * nu_a,
                delta_t * nu_a,
                dt_hlf * nu_yawdd,
                delta_t * nu_yawdd;

        // Catch zero division
        if (fabs(yaw_d) > 1e-6) {
            double vel_yaw_d = vel / yaw_d;
            double yaw_yaw_d_dt = yaw + yaw_d * delta_t;

            pred << vel_yaw_d * (sin(yaw_yaw_d_dt) - sin(yaw)),
                    vel_yaw_d * (cos(yaw) - cos(yaw_yaw_d_dt)),
                    0,
                    yaw_d * delta_t,
                    0;
        } else {
            pred << vel * cos(yaw) * delta_t,
                    vel * sin(yaw) * delta_t,
                    0,
                    0,
                    0;
        }

        Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_) + pred + noise;
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
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
}
