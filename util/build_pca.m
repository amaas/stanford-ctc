dat_out = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_full_fbank_pca/'];

featDim = 744; %dimension of data
num_files = 64;

cov_file_format=[dat_out 'covariance%d.mat'];

sigma = zeros(featDim);
numFeats = 0;

%% Get full covariance matrix for all data files
for i=1:num_files
    [tmpDat] = load(sprintf(cov_file_format,i));
    sigma = sigma+(tmpDat.sigma*tmpDat.numFeats);
    numFeats = numFeats+tmpDat.numFeats;
    fprintf('Loaded file %d\n',i);
    clear tmpDat;
end;

sigma = sigma/numFeats;

%% Find principal components                                                                            
% Get eigenvectors and eigenvalues of sigma
[V D] = eig(sigma);

% D is sorted from smallest to largest eigenvalue so pull principle
% components from back to front of V
V=fliplr(V);lambdas=diag(D);lambdas=flipud(lambdas);

% get variance explained by first 1:i eigenvalues
tot_var = cumsum(lambdas)./sum(lambdas);

% plot variance explained as a function of number of eigenvectors
% sorted in decreasing order
plot(tot_var);hold on;axis([0 744 0 1]);
xlabel('PCA dimension');ylabel(['Variance explained']);

% save eigenvectors, eigenvalues
save([dat_out 'pca.mat'],'V','lambdas');