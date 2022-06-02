clear
seed = 123456789;
rand('seed', seed);
nfolds = 5; nruns=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('C:\Users\fr\Documents\BaiduNetdiskWorkspace\PSSM_Pse_AB_DWT\Features\feature_PSSM_Pse_AB_DWT.mat');
train_datas_PSSM_Pse_AB_DWT=[[feature_PSSM_Pse_AB_DWT_7573;feature_PSSM_Pse_AB_DWT_2214],[linspace(0,0,7573),linspace(1,1,2214)]'];
test_datas_PSSM_Pse_AB_DWT=[[feature_PSSM_Pse_AB_DWT_1513;feature_PSSM_Pse_AB_DWT_319],[linspace(0,0,1513),linspace(1,1,319)]'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('C:\Users\fr\Documents\BaiduNetdiskWorkspace\PSSM_Pse_AB_DWT\Features\feature_PSSM_Pse_AB_DWT_train_balanced.mat');
train_datas_PSSM_Pse_AB_DWT_ROS=[feature_PSSM_Pse_AB_DWT_X_train_ROS,feature_PSSM_Pse_AB_DWT_y_train_ROS'];
train_datas_PSSM_Pse_AB_DWT_SMOTE=[feature_PSSM_Pse_AB_DWT_X_train_SMOTE,feature_PSSM_Pse_AB_DWT_y_train_SMOTE'];
train_datas_PSSM_Pse_AB_DWT_RUS=[feature_PSSM_Pse_AB_DWT_X_train_RUS,feature_PSSM_Pse_AB_DWT_y_train_RUS'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select dataset
% train_datas=train_datas_PSSM_Pse_AB_DWT_RUS;
% test_datas=test_datas_PSSM_Pse_AB_DWT;
train_datas=cat(2,train_datas_PSSM_Pse_AB_DWT_RUS(:,1:140),train_datas_PSSM_Pse_AB_DWT_RUS(:,end));
test_datas=cat(2,test_datas_PSSM_Pse_AB_DWT(:,1:140),test_datas_PSSM_Pse_AB_DWT(:,end));
[train_X_S,test_X_S,train_label,test_label] = map_data(train_datas,test_datas);
% train_datas=cat(2,train_datas_PSSM_Pse_AB_DWT_RUS(:,1:180),train_datas_PSSM_Pse_AB_DWT_RUS(:,321:340),train_datas_PSSM_Pse_AB_DWT_RUS(:,1461));
% test_datas=cat(2,test_datas_PSSM_Pse_AB_DWT(:,1:180),test_datas_PSSM_Pse_AB_DWT(:,321:340),test_datas_PSSM_Pse_AB_DWT(:,1461));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%做了一个line_map


% train_dim = size(train_datas,2);
% train_X = train_datas(:,1:train_dim-1);
% train_label=train_datas(:,train_dim);
% 
% test_dim = size(test_datas,2);
% test_X = test_datas(:,1:test_dim-1);
% test_label=test_datas(:,test_dim);
% 
% COM_X = [train_X;test_X];
% COM_X = line_map(COM_X);
% train_end = size(train_label,1);
% test_strat = train_end + 1;
% train_X_S = COM_X(1:train_end,:);
% test_X_S = COM_X(test_strat:end,:);
% 
% train_X_S(isnan(train_X_S)) = 0;
% test_X_S(isnan(test_X_S)) = 0;
%lammda =0.05;k_nn = 100;type = 'rbf';gamma = [2^-4];beta=0.001;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% best_result=[];
% best_AUC=0;
% for i = [20:20:620]
%     lammda = [0.4];
%     k_nn = [650];
%     gamma=[0.4];
%     beta = [0.4];
%     type = 'rbf';
%     Iterations=10;pro=2;g_nn=19;
%     now_parameters = [lammda,k_nn,gamma,beta]
%     i
%     feature_id = [1,180;181,340];
%     train_datas=cat(2,train_datas_PSSM_Pse_AB_DWT_RUS(:,1:i),train_datas_PSSM_Pse_AB_DWT_RUS(:,end));
%     test_datas=cat(2,test_datas_PSSM_Pse_AB_DWT(:,1:i),test_datas_PSSM_Pse_AB_DWT(:,end));
%     
%     [train_X_S,test_X_S,train_label,test_label] = map_data(train_datas,test_datas);
%     
%     [predict_y,distance_s,score_f] = ghknn(train_X_S,train_label,test_X_S,k_nn,lammda,gamma,beta,type);
%     
%     [ACC,SN,Spec,PE,NPV,F_score,MCC] =roc(predict_y,test_label);
%     [X,Y,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_label,score_f,1);
%     result = [SN,PE,Spec,ACC,MCC,AUC_LGC_KA]
%     if AUC_LGC_KA > best_AUC
%         best_AUC = AUC_LGC_KA;
%         best_result = result;
%         best_parameters = now_parameters;
%         best_i = i;
%     end
%     now_AUC=AUC_LGC_KA
%     best_parameters
%     best_result
%     best_i
% end




best_result=[];
best_AUC=0;
for lammda = [0.4]
    for k_nn = [650]
            for gamma=[0.4]
                for beta = [0.4]
                    type = 'rbf';
                    Iterations=10;pro=2;g_nn=19;
                    gammas = [0.4,0.4,0.4,0.4]
                    types = ["rbf","lap","liner","Poly"]
                    now_parameters = [lammda,k_nn,gamma,beta]
                    %feature_id = [1,180;181,340];
                    [predict_y,distance_s,score_f] = HG_HKNN(train_X_S,train_label,test_X_S,k_nn,lammda,gamma,beta,type);
                    [ACC,SN,Spec,PE,NPV,F_score,MCC] =roc(predict_y,test_label);
                    [X,Y,THRE,AUC_LGC_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_label,score_f,1);
                    result = [SN,PE,Spec,ACC,MCC,AUC_LGC_KA]
                    if AUC_LGC_KA > best_AUC
                        best_AUC = AUC_LGC_KA;
                        best_result = result;
                        best_parameters = now_parameters;
                    end
                    now_AUC=AUC_LGC_KA
                    best_parameters
                    best_result
                end
            end
        %end
    end
end
best_parameters
best_result


                    
function [train_X_S,test_X_S,train_label,test_label] = map_data(train_datas,test_datas)
    train_dim = size(train_datas,2);
    train_X = train_datas(:,1:train_dim-1);
    train_label=train_datas(:,train_dim);

    test_dim = size(test_datas,2);
    test_X = test_datas(:,1:test_dim-1);
    test_label=test_datas(:,test_dim);

    COM_X = [train_X;test_X];
    COM_X = line_map(COM_X);
    train_end = size(train_label,1);
    test_strat = train_end + 1;
    train_X_S = COM_X(1:train_end,:);
    test_X_S = COM_X(test_strat:end,:);

    train_X_S(isnan(train_X_S)) = 0;
    test_X_S(isnan(test_X_S)) = 0;
end
                    
%[PX, PY, AUC] = calculate_roc(predict_y,test_label);
%AUC
%figure(1);
%plot(PX,PY);
%xlabel('False positive rate');
%ylabel('True positive rate');

