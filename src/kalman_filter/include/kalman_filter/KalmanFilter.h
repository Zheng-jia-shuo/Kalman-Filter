//author Joey          time 2022.11.15          version 1.0
#include <iostream>
#include <Eigen/Dense>

class KF_ZJS
{
    public:
        KF_ZJS()
        {

        }
        
        KF_ZJS(const Eigen::VectorXd& x,const Eigen::MatrixXd& P)
        {
            x_StateVec.resize(x.rows());
            P_CovStateMat.resize(P.rows(),P.cols());

            x_StateVec = x;
            P_CovStateMat = P;
        }

        void Set_AMat(const Eigen::MatrixXd& A);
        void Set_BMat(const Eigen::MatrixXd& B);
        void Set_RMat(const Eigen::MatrixXd& R);
        void Set_QMat(const Eigen::MatrixXd& Q);
        void Set_HMat(const Eigen::MatrixXd& H);
        void Set_uVec(const Eigen::VectorXd& u);
        void Predict_P(Eigen::MatrixXd& P);
        void Update_State(Eigen::VectorXd& z,Eigen::MatrixXd& K);
        void Update_P(Eigen::MatrixXd& P,Eigen::MatrixXd& K);
        void Cal_Gain(Eigen::MatrixXd& K,Eigen::MatrixXd& P);
        Eigen::MatrixXd Get_CurrentState();
    private:
        Eigen::MatrixXd A_StateTransMat;
        Eigen::MatrixXd B_ControlMat;
        Eigen::MatrixXd Q_ProcessMat;
        Eigen::MatrixXd R_MeasureMat;
        Eigen::VectorXd u_ControlVec;
        Eigen::VectorXd x_StateVec;
        
        Eigen::MatrixXd K_KFGain;
        Eigen::MatrixXd H_MeasureMat;
        Eigen::VectorXd z_MeasureVec;

        //Pk = E[ek,ek^T]       ek = xk - xk^
        //Pk^ = E[ek^,ek^T]       ek^ = xk - xk^^
        Eigen::MatrixXd P_CovStateMat;
};