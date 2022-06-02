function [mcc_list,acc_lst,sn_list,sp_list,PE_list,rate_p_n_list] = Threhold_Find(score_s,label,min_Th,max_Th,step_Th)

max_label = max(label);
min_label = min(label);
for j=1:size(score_s,1)

	ss=score_s(j,1);
	lbb=label(j,1);
	if lbb==min_label
		score_s(j,1)=ss ;
	else
		score_s(j,1)=ss ;
	end
	
	


end


mcc_list=[];acc_lst=[];sn_list=[];sp_list=[];rate_p_n_list=[];X=[];PE_list=[];
Threshold_t=0;
for k=min_Th:step_Th:max_Th
	Threshold_t = k;
	Predict_label = zeros(size(label,1),1);
	for i=1:size(label,1)
		SC = score_s(i,1);
		if SC>Threshold_t
			PP_ = max_label;
			Predict_label(i)=PP_;
		else
			PP_ = min_label;
			Predict_label(i)=PP_;
		end
	end
	ACC=[];
	MCC=[];
	SN=[];
	SPec=[];
	PE=[];
	NPV=[];
	F_Socre=[];
	[ACC,SN,SPec,PE,NPV,F_Socre,MCC] = roc(Predict_label,label);
	rate_s = sum(find(Predict_label==max_label))/sum(find(Predict_label==min_label));
	
	PE_list=[PE_list;PE];
	mcc_list=[mcc_list;MCC];
	acc_lst=[acc_lst;ACC];
	sn_list=[sn_list;SN];
	sp_list=[sp_list;SPec];
	rate_p_n_list=[rate_p_n_list;rate_s];
	
	X=[X;k];
	

end

hold on
	%plot(X,rate_p_n_list,'k','LineWidth',1.5);
	plot(X,sn_list,'r','LineWidth',1.5);
	plot(X,sp_list,'g','LineWidth',1.5);
	plot(X,acc_lst,'y','LineWidth',1.5);
	plot(X,PE_list,'c','LineWidth',1.5);
	plot(X,mcc_list,'b','LineWidth',1.5);
	grid on;
    %ll=legend('rate p/n', 'SN', 'Spec', 'ACC','Pre', 'MCC');
    ll=legend('SN', 'Spec', 'ACC','Pre', 'MCC');
	xlabel('Threshold');ylabel('Values');
	box on;
	grid off;
	set(get(gca,'XLabel'),'FontSize',18);
	set(get(gca,'YLabel'),'FontSize',18);
	set(gca,'FontSize',10);
	set(ll,'FontSize',10);
end