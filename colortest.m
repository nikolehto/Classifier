clear;

%{
  Tarkoituksena on siis tehd� aluksi random painomatriisi kooltaan clusterien m��r� X n�ytteenvektorin pituus
    ( visualisoitu yl�s )
  
  T�m�n j�lkeen p�ivitt�� painovektoria niin kauan ett� jokaiselle random
  v�rille l�ytyy clusteri tarpeeksi l�helt� 

  Lopuksi visualisoidaan lopullinen painomatriisi ( alhaalla )
 
 Tarkoitus on siis toteuttaa t�m� http://www.ai-junkie.com/ann/som/som1.html, poikkeuksena
 toteutetaan som verkolla jossa nodet on sijoitettu vierekk�in ( kun alkuper�isess� kaksiulotteinen vektori ). Koodin pit�isi vastata
 t�t�
http://mnemstudio.org/ai/nn/som_python_ex2.txt

30.9: ohjelma k��ntyy mutta tulokset eiv�t vaikuta olevan oikeita. 
1.10 : ohjelma toimii, metodien pit�� palauttaa aina objekti
%}

learn_data_file = 'learn_data.mat';
clusters = 32;
vector_len = 3;
decay_rate = 0.96; % default 0.96
min_alpha = 0.01; % default 0.01
radius_reduction =  0.023; % default 0.023

load(learn_data_file, 'data'); 
learn_data = data; 
clear data;

%figure;
% plot 300 first samples from learn data
%Plotter(learn_data(1:300,:));

mySom = SomClass(clusters, vector_len, min_alpha, decay_rate, radius_reduction);

startWeights = mySom.mWeightArray;

mySom = mySom.training(learn_data);

figure;
% plot startWeights vs learnedWeights
Plotter(startWeights, mySom.mWeightArray);

test_data = [1.0, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0;
            0.5, 0.5, 0.5];
        
columnNames = {'Red','Green','Blue','Grey'};

for i = 1:size(test_data, 1)
    display(mat2str(test_data(i,:)));
    mySom = mySom.compute_input(test_data, i);
    minimum = mySom.get_minimum(mySom.mDeltaVector);
    display([columnNames{1,i}, ' winner is ', num2str(minimum)]);
end