//#include "kalman_filter/KalmanFilter.h"
#include "/home/zjs/git_kal/src/kalman_filter/include/kalman_filter/KalmanFilter.h"

void KF_ZJS::Set_AMat(const Eigen::MatrixXd& A)
{
    A_StateTransMat.resize(A.rows(),A.cols());
    A_StateTransMat = A;
}

void KF_ZJS::Set_BMat(const Eigen::MatrixXd& B)
{
    B_ControlMat.resize(B.rows(),B.cols());
    B_ControlMat = B;
}

void KF_ZJS::Set_RMat(const Eigen::MatrixXd& R)
{
    R_MeasureMat.resize(R.rows(),R.cols());
    R_MeasureMat = R;
}

void KF_ZJS::Set_QMat(const Eigen::MatrixXd& Q)
{
    Q_ProcessMat.resize(Q.rows(),Q.cols());
    Q_ProcessMat = Q;
}

void KF_ZJS::Set_HMat(const Eigen::MatrixXd& H)
{
    H_MeasureMat.resize(H.rows(),H.cols());
    H_MeasureMat = H;
}

void KF_ZJS::Set_uVec(const Eigen::VectorXd& u)
{
    u_ControlVec.resize(u.rows(),u.cols());
    u_ControlVec = u;
}

void KF_ZJS::Predict_P(Eigen::MatrixXd& P)
{
    P = A_StateTransMat * P_CovStateMat * A_StateTransMat.transpose() + Q_ProcessMat;
}

void KF_ZJS::Cal_Gain(Eigen::MatrixXd& K,Eigen::MatrixXd& P)
{
    P.resize(P_CovStateMat.rows(),P_CovStateMat.cols());
    K.resize(P.rows(),H_MeasureMat.rows());

    K = P * H_MeasureMat.transpose() * (H_MeasureMat * P * H_MeasureMat.transpose() + R_MeasureMat).inverse();
    K_KFGain = K;
}

void KF_ZJS::Update_State(Eigen::VectorXd& z,Eigen::MatrixXd& K)
{
    Eigen::VectorXd x_temp;
    z_MeasureVec.resize(z.rows(),z.cols());
    //K.resize(P_CovStateMat.rows(),H_MeasureMat.rows());
    x_temp.resize(x_StateVec.rows());

    z_MeasureVec = z;
    x_temp = A_StateTransMat * x_StateVec + B_ControlMat * u_ControlVec;
    x_StateVec = x_temp + K * (z_MeasureVec - H_MeasureMat * x_temp);
}

void KF_ZJS::Update_P(Eigen::MatrixXd& P,Eigen::MatrixXd& K)
{
    Eigen::MatrixXd I;
    I.resize(P.rows(),P.cols());
    // P.resize(P_CovStateMat.rows(),P_CovStateMat.cols());
    I.setIdentity();

    P_CovStateMat = (I - K * H_MeasureMat) * P;
}

Eigen::MatrixXd KF_ZJS::Get_CurrentState()
{
    return x_StateVec;
}