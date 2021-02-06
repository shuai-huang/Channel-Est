function [res, input_par, output_par] = gamp_bgm(A, y, tune_idx, gamp_par, input_par, output_par)

    % the operator version, i.e. A is an operator here

    % use MMSE formulation of GAMP with scalar variance
    % the prior distribution of the input channel is Bernoulli-Gaussian mixture
    % the prior distribution of the output channel is white-Gaussian
    M   = size(A,1);    % the dimensionality of the measurement y
    N   = size(A,2);    % the dimensionality of the signal x
    
    
    % set GAMP parameters
    max_pe_inner_ite = gamp_par.max_pe_inner_ite; % maximum number of gamp iterations
    max_pe_ite = gamp_par.max_pe_ite;   % maximum number of parameter estimation iterations
    cvg_thd = gamp_par.cvg_thd; % the convergence threshold
    kappa   = gamp_par.kappa;   % learning rate or damping rate
    eta     = gamp_par.eta;     % damping rate for the gamp iterations
    
    x_hat   = gamp_par.x_hat;   % estimated signal
    tau_x   = gamp_par.tau_x;   % signal variance
    s_hat   = gamp_par.s_hat;   % dummy variable from output function
    
    verbose = gamp_par.verbose;	% whether to output convergence values in every iteration
    
    % set input distribution parameters
    lambda  = input_par.lambda; % Bernoulli parameter
    theta   = input_par.theta;  % the means of the Gaussian mixtures
    phi     = input_par.phi;    % the variances of the Gaussian mixtures
    omega   = input_par.omega;  % the weights of the Gaussian mixtures
    num_c   = input_par.num_c;  % the number of Gaussian mixtures
    
    % set output distribution parameters
    tau_w   = output_par.tau_w; % the white-Gaussian noise variance
    
    tune_vect = de2bi(tune_idx);
    %tune_vect = [0 0 0 0 0 0 0 1];
    if (length(tune_vect)<8)
        tune_vect = [tune_vect repmat(0, 1, 8-length(tune_vect))];
    end

    tau_p = A.multSq(tau_x);
    p_hat = A.mult(x_hat) - tau_p .* s_hat;

    for (ite_pe = 1:max_pe_ite)

        x_hat_pe_pre = x_hat;
        
        % output nonlinear step
        if (ite_pe>1)
            s_hat_pre = s_hat;
            tau_s_pre = tau_s;
        end
        tau_s = 1 ./ (tau_w + tau_p);
        s_hat = (y - p_hat) .* tau_s;
        if (ite_pe>1)
            if (tune_vect(4)==1)
                s_hat = s_hat_pre + eta*(s_hat-s_hat_pre);
            end
            if (tune_vect(3)==1)
                tau_s = tau_s_pre + eta*(tau_s-tau_s_pre);
            end
        end

        tau_s(tau_s<eps) = eps;

        % input linear step
        if (ite_pe>1)
            tau_r_pre = tau_r;
            r_hat_pre = r_hat;
        end

        tau_r = 1 ./ A.multSqTr(tau_s);
	    r_hat = x_hat + tau_r .* A.multTr(s_hat);	% this is conjungate transpose
        if (ite_pe>1)
            if (tune_vect(5)==1)
                tau_r = tau_r_pre + eta*(tau_r-tau_r_pre);
            end
            if (tune_vect(6)==1)
                r_hat = r_hat_pre + eta*(r_hat-r_hat_pre);
            end
        end


        % parameter estimation
        for (ite_par = 1:max_pe_inner_ite)
            [lambda, omega, theta, phi] = input_parameter_est(r_hat, tau_r, lambda, omega, theta, phi, kappa); 
        end

        % input nonlinear step
        if (ite_pe>1)
            x_hat_pre = x_hat;
            tau_x_pre = tau_x;
        end

        [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi);
        if (ite_pe>1)
            if (tune_vect(8)==1)
                x_hat = x_hat_pre + eta*(x_hat-x_hat_pre);
            end
            if (tune_vect(7)==1)
                tau_x = tau_x_pre + eta*(tau_x-tau_x_pre);
            end
        end


        % output linear step
        if (ite_pe>1)
            tau_p_pre = tau_p;
            p_hat_pre = p_hat;
        end

        tau_p = A.multSq(tau_x);
        p_hat = A.mult(x_hat) - tau_p .* s_hat;
        if (ite_pe>1)
            if (tune_vect(1)==1)
                tau_p = tau_p_pre + eta*(tau_p-tau_p_pre);
            end
            if (tune_vect(2)==1)
                p_hat = p_hat_pre + eta*(p_hat-p_hat_pre);
            end
        end


        % parameter estimation
        for (ite_par = 1:max_pe_inner_ite)
            tau_w = output_parameter_est(y, tau_w, p_hat, tau_p, kappa);
        end

        cvg_val_pe = norm(x_hat-x_hat_pe_pre) / norm(x_hat);
        
		if (verbose)
        	fprintf('Iteration %d: %d\n', ite_pe, cvg_val_pe);
        end

        if (cvg_val_pe<cvg_thd) 
            fprintf('Convergence reached\n');
            fprintf('Lambda: %f\n', lambda);
            fprintf('Omega: ');
            for (i=1:num_c)
                fprintf('%f ', omega(i))
            end
            fprintf('\n')

            fprintf('Theta: ');
            for (i=1:num_c)
                fprintf('%f + %f i ', real(theta(i)), imag(theta(i)))
            end
            fprintf('\n')

            fprintf('Phi: ')
            for (i=1:num_c)
                fprintf('%f ', phi(i))
            end
            fprintf('\n')

            fprintf('Tau: %f\n', tau_w);
            break;
        end
    end
    
    % save the recovery results
    res.x_hat = x_hat;  % the recovered signal
    res.tau_x = tau_x;
    res.s_hat = s_hat;
    res.tau_s = tau_s;
    res.p_hat = p_hat;
    res.tau_p = tau_p;
    res.r_hat = r_hat;
    res.tau_s = tau_r;
    
    % update input distribution parameters
    input_par.lambda = lambda; % Bernoulli parameter
    input_par.theta = theta;  % the means of the Gaussian mixtures
    input_par.phi   = phi;    % the variances of the Gaussian mixtures
    input_par.omega = omega;  % the weights of the Gaussian mixtures
    
    % update output distribution parameters
    output_par.tau_w = tau_w; % the white-Gaussian noise variance

end

function [x_hat, tau_x] = input_function(r_hat, tau_r, lambda, omega, theta, phi)
    
    dim_smp=length(r_hat);
    num_cluster=length(omega);

    cluster_block_exp_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        cluster_block_exp_mat(:,i) = -(abs(theta(i)-r_hat) ./ sqrt(phi(i)+tau_r)).^2;
    end

    cluster_block_exp_max = zeros(dim_smp, 1);
    for (i=1:dim_smp)
        cluster_block_exp_max(i) = max(cluster_block_exp_mat(i,:));
    end

    block_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        block_mat(:,i) = omega(i) * (tau_r./(phi(i)+tau_r)) .* exp( -cluster_block_exp_max -(abs(theta(i)-r_hat) ./ sqrt(phi(i)+tau_r)).^2 );
    end

    % compute x_hat
    block_mat_nmr_x = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x(:,i) = block_mat_nmr_x(:,i) .* (theta(i) * tau_r + r_hat * phi(i)) ./ (phi(i) + tau_r);
    end

    nmr_x = sum(block_mat_nmr_x, 2);
    dnm_x = sum(block_mat, 2) + ((1-lambda)/lambda) * exp( -cluster_block_exp_max -(abs(r_hat) ./ sqrt(tau_r)).^2 );

    x_hat = nmr_x ./ (dnm_x);

    % compute tau_x
    block_mat_nmr_x_sq = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x_sq(:,i) = block_mat_nmr_x_sq(:,i) .* (  phi(i)*tau_r./(phi(i) + tau_r) + (abs( (theta(i)*tau_r + r_hat*phi(i)) ./ (phi(i) + tau_r) )).^2  );
    end

    nmr_x_sq = sum(block_mat_nmr_x_sq, 2);
    dnm_x_sq = dnm_x;

    tau_x = nmr_x_sq ./ dnm_x_sq - (abs(x_hat)).^2;  % this is nonnegative in theory
    tau_x = max(tau_x, 1e-12);  % just in case
    %sum(tau_x)

    %min(phi(1))
    %min(tau_r)

end

function [s_hat, tau_s] = output_function(p_hat, tau_p, y_real_lower, y_real_upper, y_imag_lower, y_imag_upper, tau_w)

    % divide tau_w and tau_p by 2
    tau_w = 0.5*tau_w;
    tau_p = 0.5*tau_p;

    p_hat_real = real(p_hat);
    p_hat_imag = imag(p_hat);
    % so many complicated steps just to prevent NAN, why doesn't MATLAb have a clever way to compute 0*Inf???	
    p_bar = p_hat./sqrt(tau_w+tau_p);    % should this be divided by 2
    p_real_bar = real(p_bar);
    p_imag_bar = imag(p_bar);

    %sum(sqrt(tau_w+tau_p))
    %min(tau_w)
    %min(tau_p)

    y_real_lower_bar = y_real_lower./sqrt(tau_w+tau_p);
    y_real_upper_bar = y_real_upper./sqrt(tau_w+tau_p);
    y_real_upper_bar_minus_p_bar = y_real_upper_bar-p_real_bar;
    y_real_lower_bar_minus_p_bar = y_real_lower_bar-p_real_bar;
    exp_y_real_upper_bar_minus_p_bar = exp(-0.5*(y_real_upper_bar_minus_p_bar).^2);
    exp_y_real_lower_bar_minus_p_bar = exp(-0.5*(y_real_lower_bar_minus_p_bar).^2);
    mut_exp_y_real_upper_bar_minus_p_bar = (y_real_upper_bar_minus_p_bar).*exp_y_real_upper_bar_minus_p_bar;
    mut_exp_y_real_upper_bar_minus_p_bar((abs(y_real_upper_bar_minus_p_bar)==Inf)&(exp_y_real_upper_bar_minus_p_bar==0))=0;
    mut_exp_y_real_upper_bar_minus_p_bar((y_real_upper_bar_minus_p_bar==0)&(abs(exp_y_real_upper_bar_minus_p_bar)==Inf))=0;
    mut_exp_y_real_lower_bar_minus_p_bar = (y_real_lower_bar_minus_p_bar).*exp_y_real_lower_bar_minus_p_bar;
    mut_exp_y_real_lower_bar_minus_p_bar((abs(y_real_lower_bar_minus_p_bar)==Inf)&(exp_y_real_lower_bar_minus_p_bar==0))=0;
    mut_exp_y_real_lower_bar_minus_p_bar((y_real_lower_bar_minus_p_bar==0)&(abs(exp_y_real_lower_bar_minus_p_bar)==Inf))=0;


    exp_diff_real_block = exp_y_real_upper_bar_minus_p_bar - exp_y_real_lower_bar_minus_p_bar;
    exp_diff_real_block_mod = mut_exp_y_real_upper_bar_minus_p_bar - mut_exp_y_real_lower_bar_minus_p_bar;

    %sum(y_real_upper_bar_minus_p_bar)
    %sum(y_real_lower_bar_minus_p_bar)

    PI_0_real = 0.5*(erf(sqrt(0.5)*(y_real_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_real_lower_bar_minus_p_bar))); % to improve the precision
    PI_0_real(abs(PI_0_real)<eps) = eps;

    PI_1_real = p_hat_real.*PI_0_real - tau_p./sqrt(2*pi*(tau_w+tau_p)).*exp_diff_real_block;

    PI_2_real = (p_hat_real.^2+tau_p).*PI_0_real - (1/sqrt(2*pi))./(tau_w+tau_p).*(tau_p.^2).*exp_diff_real_block_mod - 2/sqrt(2*pi)*tau_p.*p_real_bar.*exp_diff_real_block;

    z_hat_real = PI_1_real./PI_0_real;
    z_hat_sq_real = PI_2_real./PI_0_real;


    y_imag_lower_bar = y_imag_lower./sqrt(tau_w+tau_p);
    y_imag_upper_bar = y_imag_upper./sqrt(tau_w+tau_p);
    y_imag_upper_bar_minus_p_bar = y_imag_upper_bar-p_imag_bar;
    y_imag_lower_bar_minus_p_bar = y_imag_lower_bar-p_imag_bar;
    exp_y_imag_upper_bar_minus_p_bar = exp(-0.5*(y_imag_upper_bar_minus_p_bar).^2);
    exp_y_imag_lower_bar_minus_p_bar = exp(-0.5*(y_imag_lower_bar_minus_p_bar).^2);
    mut_exp_y_imag_upper_bar_minus_p_bar = (y_imag_upper_bar_minus_p_bar).*exp_y_imag_upper_bar_minus_p_bar;
    mut_exp_y_imag_upper_bar_minus_p_bar((abs(y_imag_upper_bar_minus_p_bar)==Inf)&(exp_y_imag_upper_bar_minus_p_bar==0))=0;
    mut_exp_y_imag_upper_bar_minus_p_bar((y_imag_upper_bar_minus_p_bar==0)&(abs(exp_y_imag_upper_bar_minus_p_bar)==Inf))=0;
    mut_exp_y_imag_lower_bar_minus_p_bar = (y_imag_lower_bar_minus_p_bar).*exp_y_imag_lower_bar_minus_p_bar;
    mut_exp_y_imag_lower_bar_minus_p_bar((abs(y_imag_lower_bar_minus_p_bar)==Inf)&(exp_y_imag_lower_bar_minus_p_bar==0))=0;
    mut_exp_y_imag_lower_bar_minus_p_bar((y_imag_lower_bar_minus_p_bar==0)&(abs(exp_y_imag_lower_bar_minus_p_bar)==Inf))=0;


    exp_diff_imag_block = exp_y_imag_upper_bar_minus_p_bar - exp_y_imag_lower_bar_minus_p_bar;
    exp_diff_imag_block_mod = mut_exp_y_imag_upper_bar_minus_p_bar - mut_exp_y_imag_lower_bar_minus_p_bar;

    PI_0_imag = 0.5*(erf(sqrt(0.5)*(y_imag_upper_bar_minus_p_bar))-erf(sqrt(0.5)*(y_imag_lower_bar_minus_p_bar))); % to improve the precision
    PI_0_imag(abs(PI_0_imag)<eps) = eps;

    PI_1_imag = p_hat_imag.*PI_0_imag - tau_p./sqrt(2*pi*(tau_w+tau_p)).*exp_diff_imag_block;

    PI_2_imag = (p_hat_imag.^2+tau_p).*PI_0_imag - (1/sqrt(2*pi))./(tau_w+tau_p).*(tau_p.^2).*exp_diff_imag_block_mod - 2/sqrt(2*pi)*tau_p.*p_imag_bar.*exp_diff_imag_block;

    z_hat_imag = PI_1_imag./PI_0_imag;
    z_hat_sq_imag = PI_2_imag./PI_0_imag;

    z_hat = complex(z_hat_real, z_hat_imag);
    z_hat_sq = z_hat_sq_real + z_hat_sq_imag;


    tau_z = z_hat_sq-abs(z_hat).^2;

    % multiply tau_w and tau_p by 2
    tau_w = 2*tau_w;
    tau_p = 2*tau_p;

    s_hat = 1./tau_p.*(z_hat-p_hat);
    tau_s = 1./tau_p.*(1-tau_z./tau_p);

end

function [lambda, omega, theta, phi] = input_parameter_est(r_hat, tau_r, lambda, omega, theta, phi, kappa)
   
    lambda_pre = lambda;
    omega_pre = omega;
    theta_pre = theta;
    phi_pre = phi;

    % do we need some kind of normalization here?
    dim_smp=length(r_hat);
    num_cluster=length(omega);

    cluster_block_exp_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        cluster_block_exp_mat(:,i) = -(abs(theta(i)-r_hat) ./ sqrt(phi(i)+tau_r)).^2;
    end

    cluster_block_exp_max = zeros(dim_smp, 1);
    for (i=1:dim_smp)
        cluster_block_exp_max(i) = max(cluster_block_exp_mat(i,:));
    end

    lambda_tmp_mat_1 = zeros(dim_smp, num_cluster);
    for (i = 1:num_cluster)
        lambda_tmp_mat_1(:,i) = lambda * omega(i) * 1./(tau_r+phi(i)) .* exp(-cluster_block_exp_max - (abs(r_hat-theta(i)) ./ sqrt(tau_r+phi(i))).^2 );
    end
    %sum(lambda_tmp_mat_1)

    % compute lambda
    lambda_tmp_1 = sum(lambda_tmp_mat_1, 2);
    lambda_tmp_2 = (1-lambda) * 1 ./ tau_r .* exp(-cluster_block_exp_max - (abs(r_hat) ./ sqrt(tau_r)).^2);

    lambda_block_2 = lambda_tmp_2 ./ (lambda_tmp_1 + lambda_tmp_2);
    lambda_block_1 = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        lambda_block_1(:,i) = lambda_tmp_mat_1(:,i) ./ (lambda_tmp_1 + lambda_tmp_2);
    end
    %sum(lambda_block_1)

    lambda_new = sum(sum(lambda_block_1)) / (sum(lambda_block_2) + sum(sum(lambda_block_1)));
    lambda = lambda + kappa * (lambda_new - lambda);

    
    % compute omega
    omega_new = sum(lambda_block_1);
    omega_new = omega_new.';	% this is transpose not conjungate transpose
    omega_new = omega_new / sum(omega_new);
    
    omega = omega + kappa * (omega_new - omega);
    
    % compute theta
    theta_tmp_mat = lambda_tmp_mat_1;
    theta_tmp_mat_sum = sum(theta_tmp_mat, 2);
    for (i=1:num_cluster)
        theta_tmp_mat(:,i) = theta_tmp_mat(:,i) ./ ( lambda_tmp_2 + theta_tmp_mat_sum );
    end
    %sum(theta_tmp_mat)
    
    theta_tmp_mat_1 = theta_tmp_mat;
    theta_tmp_mat_2 = theta_tmp_mat;
    for (i=1:num_cluster)
        theta_tmp_mat_1(:,i) = theta_tmp_mat_1(:,i) .* ( r_hat ./ (phi(i)+tau_r) );
        theta_tmp_mat_2(:,i) = theta_tmp_mat_2(:,i) ./ (phi(i)+tau_r);
    end
    theta_tmp_mat_2_sum = sum(theta_tmp_mat_2);
    theta_tmp_mat_2_sum(theta_tmp_mat_2_sum==0) = eps;
    theta_new = sum(theta_tmp_mat_1) ./ theta_tmp_mat_2_sum;   % to avoid division by 0
    theta_new = theta_new.';	% this is transpose, not conjungate transpose
    
    theta = theta + kappa * (theta_new - theta);
    
    % compute phi
    der_phi_tmp_mat_1 = theta_tmp_mat;
    der_phi_tmp_mat_2 = theta_tmp_mat;
    for (i=1:num_cluster)
        der_phi_tmp_mat_1(:,i) = der_phi_tmp_mat_1(:,i) .* ((abs((r_hat-theta(i))./(phi(i)+tau_r))).^2 - 1./(phi(i)+tau_r));
        der_phi_tmp_mat_2(:,i) = der_phi_tmp_mat_2(:,i) ./ (phi(i)+tau_r) .* ( -2*(abs((r_hat-theta(i))./(phi(i)+tau_r))).^2 + 1./(phi(i)+tau_r) );
    end

    phi_new = [];
    for (i=1:num_cluster)
        phi_new_tmp = 0;
        if (sum(der_phi_tmp_mat_2(:,i))<0)
            phi_new_tmp = phi(i) - sum(der_phi_tmp_mat_1(:,i))/sum(der_phi_tmp_mat_2(:,i));
        else
            if (sum(der_phi_tmp_mat_1(:,i))>0)
                phi_new_tmp = phi(i)*1.1;
            else
                phi_new_tmp = phi(i)*0.9;
            end
        end
        phi_new = [phi_new phi_new_tmp];
    end
    phi_new = phi_new.';	% this is transpose, not conjungate transpose

    for (i=1:num_cluster)
        %phi_new(i) = max(phi_new(i), 1e-12);    % make sure the variance is non-negative
        if (phi_new(i)<0)
            phi_new(i) = phi(i);
        end
    end
    
    phi = phi + kappa * (phi_new - phi);

end

function tau_w = output_parameter_est(y, tau_w, p_hat, tau_p, kappa)

    % divide the tau_p and tau_w by 2 here

    tau_w_new = max(mean( (abs(p_hat-y)).^2 - tau_p ), 1e-12);;

    tau_w = tau_w + kappa * (tau_w_new - tau_w);

    tau_w = max(tau_w, 1e-12);
end
