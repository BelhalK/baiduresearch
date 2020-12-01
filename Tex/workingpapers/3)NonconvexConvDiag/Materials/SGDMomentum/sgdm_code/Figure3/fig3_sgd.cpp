#include <RcppArmadillo.h>
//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_int_distribution.hpp>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
using namespace Rcpp;

double MSE(arma::rowvec a, arma::rowvec b) {
	return arma::mean( arma::pow((a-b), 2) );
}

/* For gaussian glm */
arma::rowvec identity(arma::rowvec predictor) {
	return predictor;
}

/* For binomial glm */
arma::rowvec expit(arma::rowvec predictor) {
	// copy predictor
	arma::rowvec pred_copy = predictor;
	
	// threshold values
	pred_copy(arma::find(predictor > 60)).fill(60);
	pred_copy(arma::find(predictor < -60)).fill(-60);
	//pred_copy(arma::find(predictor < -5)).fill(-5);

	return arma::exp(pred_copy) / (1 + arma::exp(pred_copy));
}
	/*
	if (arma::max(predictor) > 60) {
		return arma::ones<arma::rowvec>(predictor.size());
	} else if (arma::min(predictor) < -60) {
		return arma::zeros<arma::rowvec>(predictor.size());
	} else {
		return arma::exp(predictor) / (1 + arma::exp(predictor));
	}
	*/


/* Defines struct to hold all operational variables*/
struct param_struct {
	struct {
		double gamma;
		double beta;
		int niter_batch;
		int batch_size;
		int nepoch;
		int min_burnin;
		bool mom_switch_bool;
		double mom_switch_threshold;
		double final_beta;
		bool constLR;
		bool autoLR;
		int eval_interval;
	} args; // Declares this struct

	struct {
		arma::rowvec theta;
	  arma::rowvec theta_prev;
	  arma::rowvec grad_prev;
		double lr;
		double beta;
		NumericVector burnin;
		NumericVector convg;
		int burnin_idx;
		int convg_idx;
		bool burnin_done;
		bool convg_done;
		arma::rowvec momentum;
		int mom_switch;
		bool mom_switch_done;
	} var;

	struct {
		arma::rowvec train_loss;
		arma::rowvec test_loss;
		arma::rowvec train_acc;
		arma::rowvec test_acc;
		//NumericVector train_loss;
		//NumericVector test_loss;
		//NumericVector train_acc;
		//NumericVector test_acc;
		arma::rowvec ip_all; // Need slice
		arma::rowvec grad_norm_all;
		arma::rowvec mse_step;
		//NumericVector grad_norm_all;
		//NumericVector mse_step;
	} out;
};

List struct_to_list(param_struct *params) {
  List param_list = List::create(
    Named("args") = List::create(
      Named("gamma")                = params->args.gamma,
      Named("beta")                 = params->args.beta,
      Named("niter_batch")          = params->args.niter_batch,
      Named("batch_size")           = params->args.batch_size,
      Named("nepoch")               = params->args.nepoch,
      Named("min_burnin")           = params->args.min_burnin,
      Named("mom_switch_bool")      = params->args.mom_switch_bool,
      Named("mom_switch_threshold") = params->args.mom_switch_threshold,
      Named("final_beta")           = params->args.final_beta,
      Named("constLR")              = params->args.constLR,
      Named("autoLR")               = params->args.autoLR,
			Named("eval_interval")        = params->args.eval_interval
    ),
    Named("var")  = List::create(
      Named("theta")           = params->var.theta,
      Named("theta_prev")      = params->var.theta_prev,
      Named("grad_prev")       = params->var.grad_prev,
      Named("lr")              = params->var.lr,
      Named("beta")            = params->var.beta,
      Named("burnin")          = params->var.burnin,
      Named("convg")           = params->var.convg,
      Named("burnin_idx")      = params->var.burnin_idx,
      Named("convg_idx")       = params->var.convg_idx,
      Named("burnin_done")     = params->var.burnin_done,
      Named("convg_done")      = params->var.convg_done,
      Named("momentum")        = params->var.momentum,
      Named("mom_switch")      = params->var.mom_switch,
      Named("mom_switch_done") = params->var.mom_switch_done
    ),
    Named("out")  = List::create(
      Named("train_loss")    = params->out.train_loss,
      Named("test_loss")     = params->out.test_loss,
      Named("train_acc")     = params->out.train_acc,
      Named("test_accc")     = params->out.test_acc,
      Named("ip_all")        = params->out.ip_all,
      Named("grad_norm_all") = params->out.grad_norm_all,
      Named("mse_step")      = params->out.mse_step
    )
  );
  
  return param_list;
};

param_struct init_params(
		arma::mat *trainX, 
		double gamma,
		int nepoch,
		double beta,
		int batch_size,
		double burnin_frac,
		bool constLR,
		bool momentum_switch,
		bool autoLR
		) {

	/* dimension parameters */
	int N = trainX->n_rows;
	int p = trainX->n_cols;

	/* declare instance of struct */
	param_struct params;

	/* input arguments */
	params.args.gamma                = gamma;
	Rcout << "gamma: " << params.args.gamma << "\n";
	params.args.beta                 = beta;
	params.args.niter_batch          = (int) N / batch_size;
	params.args.batch_size           = batch_size;
	params.args.nepoch               = nepoch;
	params.args.min_burnin           = params.args.niter_batch * burnin_frac;
	params.args.mom_switch_bool      = momentum_switch;
	params.args.mom_switch_threshold = 5e-3;
	params.args.final_beta           = 0.2;
	params.args.constLR              = constLR;
	params.args.autoLR               = autoLR;
	params.args.eval_interval        = params.args.niter_batch / 600;

	/* intermediate variables */
	//params.var.theta                 = arma::zeros<arma::rowvec>(p); //arma::randn(p).t();
	params.var.theta                 = arma::randn(p).t();
	params.var.theta_prev            = arma::zeros<arma::rowvec>(p);
	params.var.grad_prev             = arma::zeros<arma::rowvec>(p);
	params.var.lr                    = gamma;
	params.var.beta                  = beta;
	params.var.burnin                = NumericVector(100);	
	params.var.convg                 = NumericVector(100);
	params.var.burnin_idx            = 0;
	params.var.convg_idx             = 0;
	params.var.burnin_done           = false;
	params.var.convg_done            = false;
	params.var.momentum              = arma::zeros<arma::rowvec>(p);
	params.var.mom_switch            = 0;
	params.var.mom_switch_done       = false;

	/* logging output */
	int vec_long  = (int) params.args.niter_batch * nepoch;
	int vec_short = (int) vec_long / params.args.eval_interval; 
	/* record every params.args.eval_interval interations */
	params.out.train_loss            = arma::zeros<arma::rowvec>(vec_short);
	params.out.test_loss             = arma::zeros<arma::rowvec>(vec_short);
	params.out.train_acc             = arma::zeros<arma::rowvec>(vec_short);
	params.out.test_acc              = arma::zeros<arma::rowvec>(vec_short);
	//params.out.train_loss            = NumericVector(vec_short);
	//params.out.test_loss             = NumericVector(vec_short);
	//params.out.train_acc             = NumericVector(vec_short);
	//params.out.test_acc              = NumericVector(vec_short);
	
	/* record every iteration */
	params.out.ip_all                = arma::zeros<arma::rowvec>(vec_long); //Need slice view
	params.out.grad_norm_all         = arma::zeros<arma::rowvec>(vec_long);
	params.out.mse_step              = arma::zeros<arma::rowvec>(vec_long);
	//params.out.grad_norm_all         = NumericVector(vec_long);
	//params.out.mse_step              = NumericVector(vec_long);

	return params;
}

void batch_update(
		int j,
		int epoch,
		arma::rowvec *gradn,
		param_struct *params
		) {

	/* check for decr learning rate */
	if( !params->args.constLR ) {
	  params->var.lr = params->args.gamma / j;  
	}
	
	/* momentum update */
	params->var.momentum = params->var.beta * params->var.momentum + params->var.lr * (*gradn) / params->args.batch_size;

	/* parameter update */
	params->var.theta    = params->var.theta - params->var.momentum;
	
	/* ip update */
	params->out.ip_all(j-1) = arma::dot(*gradn, params->var.grad_prev);

	/* misc update */
	params->out.grad_norm_all(j-1) = arma::norm(*gradn, 2);
	params->out.mse_step(j-1) = MSE(params->var.theta, params->var.theta_prev);

	/* update prev variables */
	params->var.theta_prev = params->var.theta;
	params->var.grad_prev  = *gradn;
	
	int last_burnin = params->var.burnin(std::max(0, params->var.burnin_idx-1));
	int start = std::max(last_burnin, j-21);
	
	/* momentum switch */
	if (j > params->args.min_burnin && params->args.mom_switch_bool==true 
					&& params->var.mom_switch_done==false) {
		if (params->out.mse_step(j-1) < params->args.mom_switch_threshold) {
			params->var.beta            = params->args.final_beta;
		  params->var.mom_switch      = j-1;
			params->var.mom_switch_done = true;
			Rcout << "Momentum reduced from " << params->args.beta << " to " << params->args.final_beta << " at iterate " << j-1 << " epoch " << (double) (j-1) / params->args.niter_batch << "\n";	
		}
	} 
	/* burnin */
	else if (j > std::max(params->args.min_burnin, last_burnin)+250 && params->var.burnin_done==false && 
			arma::mean(params->out.ip_all.subvec(start,j-1)) < 0) {
		params->var.burnin(params->var.burnin_idx) = j-1;
		params->var.burnin_idx                    += 1;
		params->var.burnin_done                    = true;
		Rcout << "Burnin done at iterate:" << j-1 << ", epoch:" << (double) (j-1)/ params->args.niter_batch << "\n";
	}	
	/* convergence diagnostic */
	else if (j > last_burnin+250 && params->var.burnin_done==true && params->var.convg_done==false && params->var.lr > 1.0e-5) {
		//double ip_sum = arma::sum(params->out.ip_all.tail((int) params->args.niter_batch * params->args.nepoch - 
		//			params->var.burnin(params->var.burnin_idx-1)));
		
		/* Best to wait a bit for stable estimate */
		double ip_sum2 = arma::sum(params->out.ip_all.subvec(last_burnin, j-1));

		/* convergence condition */
		if (ip_sum2 < 0) {
			params->var.convg(params->var.convg_idx) = j-1;
			params->var.convg_done                   = true;
			Rcout << "Convergence diagnostic activated at iterate:" << j-1 << 
				", epoch:" << (double) (j-1)/ params->args.niter_batch << "with ip sum: " << ip_sum2 <<"\n";
		}
	}
	/* repeat LR */
	else if ( (params->args.autoLR==true) && (params->var.convg_done==true) && (params->var.lr > 1.0e-5) ) {
			params->var.lr         *= 0.10;
			params->var.burnin_done = false;
			params->var.convg_done  = false;
			Rcout << "Learning rate reduced x0.1 to " << params->var.lr << " at iterate: " <<
				j-1 << " epoch: " << (double) (j-1) / params->args.niter_batch << "\n";
			Rcout << "condition check: " << (params->var.lr > 1.0e-5) << "\n";
	}
}

void eval_update(
		arma::mat *trainX,
		arma::mat *testX,
		arma::rowvec *trainY,
		arma::rowvec *testY,
		int j,
		param_struct *params,
		arma::rowvec (*glm_link_loc) (arma::rowvec)
		) {

	/* loss update */
	arma::rowvec pred_train = glm_link_loc(params->var.theta * (*trainX).t());
	arma::rowvec pred_test = glm_link_loc(params->var.theta * (*testX).t());

	params->out.train_loss((j-1)/params->args.eval_interval) = 
		arma::sum((*trainY) % arma::log(pred_train) + (1 - *trainY) % arma::log(1 - pred_train + 1e-25)); 

	params->out.test_loss((j-1)/params->args.eval_interval) = 
		arma::sum((*testY) % arma::log(pred_test) + (1 - *testY) % arma::log(1 - pred_test + 1e-25)); 

	/* acc update */
	arma::rowvec class_train = arma::zeros<arma::rowvec>(pred_train.size());
	arma::rowvec class_test = arma::zeros<arma::rowvec>(pred_test.size());
	
	class_train(arma::find(pred_train >= 0.5)).ones();
	class_test(arma::find(pred_test >= 0.5)).ones();
	
	params->out.train_acc((j-1)/params->args.eval_interval) = 
		1 - arma::mean( arma::abs(*trainY - class_train) );
	params->out.test_acc((j-1)/params->args.eval_interval)  = 
	  1 - arma::mean( arma::abs(*testY - class_test) );
}


// [[Rcpp::export]]
List momentum_sgd_Cpp(
		arma::mat trainX,
		arma::mat testX,
		arma::rowvec trainY,
		arma::rowvec testY,
		std::string model_name,
		double gamma,
		int nepoch,
		double beta,
		int batch_size,
		double burnin,
		bool constLR,
		bool momentum_switch,
		bool autoLR,
		uint32_t seed
		) {

	/* initialize parameters */
	param_struct params = init_params(
	    &trainX,
			gamma,
			nepoch,
			beta,
			batch_size,
			burnin,
			constLR,
			momentum_switch,
			autoLR);

	/* dimension parameters */
	int N = trainX.n_rows;
	int p = trainX.n_cols;

	/* choose glm_link */
	arma::rowvec (*glm_link_loc) (arma::rowvec);
	if (model_name.compare("gaussian") == 0) {
		glm_link_loc = identity;
	} else if (model_name.compare("binomial") == 0) {
		glm_link_loc = expit;
	} else {
		throw std::invalid_argument("model_name not gaussian or binomial");
	}

	/* random index for mini-batch */
	arma::urowvec idx; 

	/* main loop */
	int j = 1;
	arma::rowvec grad_minibatch = arma::zeros<arma::rowvec>(p);
	arma::mat x_n               = arma::zeros<arma::mat>(params.args.batch_size, p);
	arma::rowvec pred_n         = arma::zeros<arma::rowvec>(params.args.batch_size);
	arma::rowvec y_n            = arma::zeros<arma::rowvec>(params.args.batch_size);

	/* epochs */
	for (int epoch=0; epoch < params.args.nepoch; epoch++) {
		/* iterates */
		for (int n=0; n < params.args.niter_batch; n++) {
			/* minibatch vectorized */
			idx    = arma::randi<arma::urowvec>(params.args.batch_size, arma::distr_param(0,N-1));
			x_n    = trainX.rows(idx);
			y_n    = trainY(idx).t();
			pred_n = params.var.theta * x_n.t();
			grad_minibatch = arma::mean( -(y_n - glm_link_loc(pred_n)).t() % x_n.each_col(), 0);
			//Rcout << grad_minibatch << "\n";
			/*
			Rcout << "grad norm: " << arma::norm(grad_minibatch, 2) << "\n";
			Rcout << "pred norm: " << arma::norm(pred_n, 2) << "\n";
			Rcout << "pred shape: " << arma::size(pred_n) << "\n";
			Rcout << "h() norm: " << arma::norm(glm_link_loc(pred_n), 2) << "\n";
			Rcout << "h() shape: " << arma::size(glm_link_loc(pred_n)) << "\n";
			Rcout << "y_n norm: " << arma::norm(y_n, 2) << "\n";
			Rcout << "y_n shape: " << arma::size(y_n) << "\n";
			Rcout << "x_n norm: " << arma::norm(x_n, 2) << "\n";
			Rcout << "x_n shape: " << arma::size(x_n) << "\n";
			Rcout << "part 1 norm: " << arma::norm((y_n - glm_link_loc(pred_n)), 2) << "\n";
			Rcout << "part 1 size: " << arma::size(y_n - glm_link_loc(pred_n)) << "\n";
			*/


			/* minibatch update */
			batch_update(j, epoch, &grad_minibatch, &params);
			j += 1;

			/* update loss/acc */
			if (j % params.args.eval_interval == 0) {
				eval_update(&trainX, &testX, &trainY, &testY, j, &params, glm_link_loc);
			}

			/* realtime update */
			if (j % 1000 == 0) {
				Rcout << "index: " << j << " percentage: " << 
					(double) j / (params.args.niter_batch * params.args.nepoch) << "\n";;
			}
		}	
	}	
	
	// Convert struct to List
	List param_list = struct_to_list(&params);

	return param_list;
}
