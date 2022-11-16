#include <Eigen/Dense>
#include "/home/zjs/git_kal/src/kalman_filter/include/kalman_filter/KalmanFilter.h"

const double g = 9.8;
#define ARR_NUM 20

int main(int argc,char* argv[])
{
    Eigen::VectorXd x_state;
    Eigen::MatrixXd P_Cov;
    Eigen::MatrixXd A_Mat;
    Eigen::MatrixXd B_Mat;
    Eigen::MatrixXd H_Mat;
    Eigen::MatrixXd Q_Mat;
    Eigen::MatrixXd R_Mat;
    Eigen::VectorXd u_Vec;

    x_state.resize(2);
    P_Cov.resize(2,2);
    A_Mat.resize(2,2);
    B_Mat.resize(2,2);
    H_Mat.resize(1,2);
    Q_Mat.resize(2,2);
    R_Mat.resize(1,1);
    u_Vec.resize(2);

    x_state(0) = 1900;
    x_state(1) = 10;
    P_Cov.setZero();
    P_Cov(0,0) = 100;
    P_Cov(1,1) = 2;
    A_Mat.setIdentity();
    B_Mat.setIdentity();
    H_Mat.setIdentity();
    Q_Mat.setZero();
    R_Mat.setIdentity();
    u_Vec(0) = -0.5 * g;
    u_Vec(1) = g;
    // std::cout<<"H_Mat = "<<H_Mat<<std::endl;

    KF_ZJS kf(x_state,P_Cov);
    kf.Set_AMat(A_Mat);
    kf.Set_BMat(B_Mat);
    kf.Set_HMat(H_Mat);
    kf.Set_QMat(Q_Mat);
    kf.Set_RMat(R_Mat);
    kf.Set_uVec(u_Vec);

    double arr[ARR_NUM]={1994.5,1979.4,1955.4,1921.4,1877.7,1825.0,1759.8,1686.7,1603.6,1509.2,
                                                          1407.6,1294.4,1172.4,1039.9,898.0,745.5,585.0,412.5,231.8,39.9};

    for(int i=0; i<20; i++)
    {
        Eigen::VectorXd z;
        z.resize(1);
        z(0) = arr[i];

        Eigen::MatrixXd P_Pre;
        P_Pre.resize(P_Cov.rows(),P_Cov.cols());

        Eigen::MatrixXd K_m;
        K_m.resize(P_Cov.rows(),H_Mat.rows()); 

        kf.Predict_P(P_Pre);
        kf.Cal_Gain(K_m,P_Pre);
        kf.Update_State(z,K_m);
        kf.Update_P(P_Pre,K_m);

        std::cout<<kf.Get_CurrentState()<<std::endl;
        std::cout<<"----------------"<<std::endl;
    }

    return 0;
}