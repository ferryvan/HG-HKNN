function [predict_y,distance_s,score_f] = HG_HKNN(train_x,train_y,test_x,k_nn,lammda,gamma,beta,type)


Trainlabels=train_y;
uniqlabels=unique(train_y);
c=max(size(uniqlabels));
n_test = size(test_x,1);

distance_s = zeros(n_test,c);

number_c = zeros(c,1);

for i=1:c
	train_x_c = train_x(find(Trainlabels==uniqlabels(i)),:); %train_x_c为c类别的训练样本
	n_c= size(train_x_c,1);number_c(i) = n_c; %n_c为c类别的训练样本个数
	if k_nn>=n_c
		k_nn_i = n_c;
	else
		k_nn_i = k_nn;
	end
	[k_nn_x_id] = Nearest_Neighbor(k_nn_i,train_x_c,test_x); 
   for j=1:n_test
        %fprintf("%d,%d\n",i,j)
		N_x = train_x_c(k_nn_x_id(j,:),:);  %k*d矩阵
		N_mu = mean(N_x);             %d维列向量，每个值为每个特征的均值
		V = N_x-repmat(N_mu,size(N_x,1),1);  % k*d矩阵 ，每行与对应特征平均值相减之后的样本特征
        
        % 准备各个式子
		K_n_tr = kernel_function(V,V,gamma,type); % k*k核矩阵
		nc_x = test_x(j,:)-N_mu;   %第j个测试集样本减去特征的均值向量
		K_n_tr_1 = kernel_function(V,nc_x,gamma,type); %k*1向量
        
        L_M = construct_Hypergraphs_knn(K_n_tr,2);
		%L_M = Lap_M_computing(K_n_tr); %计算K(Vc,Vc)的拉普拉斯矩阵
         
        %计算K(x,x)
        K_x_x = kernel_function(nc_x,nc_x,gamma,type); %计算K(x,x)
         
		alpha = (K_n_tr + lammda*eye(k_nn_i) + beta*L_M)\(K_n_tr_1);  
        % 计算c类的alpha为k*1 注意这里用左除代替了矩阵取逆
        
        dis = sqrt(K_x_x - 2*K_n_tr_1'*alpha + alpha'*K_n_tr*alpha);
		distance_s(j,i) = real(dis);% 第j个样本到第i类的距离 负数开方后直接取0
   end
   
end

[maxval,indices]=min(distance_s');
% 取出最大项并得到序号
predict_y=uniqlabels(indices); %考虑多个最近相同大小的情况
	sum_ss=distance_s(:,1)+distance_s(:,2);   
    %只取前两列，可能是因为只有在2分类时F_score才有意义
	score_f=distance_s(:,1)./sum_ss;
end


function [k_nn_x_id] = Nearest_Neighbor(k,xtr,xte) %取欧式距离K最近

[n l] = size(xte);
k_nn_x_id = zeros(n,k);

[r] = euclidean_d(xte,xtr);

	for i=1:size(r,1)
		
		 [c ii]=sort(r(i,:));
		k_nn_x_id(i,:)=ii(1:k);  %k_nn_x_id中i行为测试集中第i个样本到训练集中最近的K个样本的序号
	end


end

function [r2] = euclidean_d(X,Y)
    % X:size(X,1)*N  Y:size(Y,1)*M
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	r2 = sqrt(r2); % r2矩阵（X*Y）中i行j列为X中i行与Y中j行的欧氏距离
end


function k= kernel_function(X,Y,gamma,type)

if strcmp(type, 'rbf')
	k = kernel_RBF(X,Y,gamma);
elseif strcmp(type,'lap')
	k = kernel_Laplace(X,Y,gamma);
elseif strcmp(type,'liner')
	k = kernel_Liner(X,Y);
elseif strcmp(type,'Poly')
	k = kernel_Polynomial(X,Y,gamma);
end


end

%RBF kernel function
function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); % RBF核矩阵
end


%Liner kernel function
function k = kernel_Liner(X,Y)

	k = X*Y'; % 核矩阵
end


%Polynomial kernel function
function k = kernel_Polynomial(X,Y,gamma)
	coef = 0.01;d=2.0;
	k =  (X*Y'*gamma + coef).^d; % 核矩阵
end


%Laplace kernel function
function k = kernel_Laplace(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	r = sqrt(r2);
	k = exp(-r*gamma); % 核矩阵
end
 
function L_M = Lap_M_computing(SS)  %计算K(Vc,Vc)的拉普拉斯矩阵

num_2 = size(SS,1);

L_M=[];
d = sum(SS);
D = diag(d);
L_D = D - SS;
%Laplacian Regularized
d_tmep=eye(num_2)/(D^(1/2));
L_M = d_tmep*L_D*d_tmep;

end