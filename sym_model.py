# Symmetric Neural network model used for experiments

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy, os

import torch.nn as nn
import torch.optim as optim

print("pytorch version ", torch.__version__)
print("Is cuda available? ", torch.cuda.is_available())



# this is a custom module for the teacher
# the weights lie randomly on a unit (d-1)-ball manifold
class Teacher_sphere(nn.Module):
    def __init__(self, dim, H_T, orthogonal_weights = False):
        """
        Instantiate weights radomly on a (d-1)-ball manifold
        """
        super().__init__()


        if not(orthogonal_weights):
          # we normalize a gaussian weight to make it a unit vector

          weights = torch.randn(dim, H_T)
          weights /= torch.linalg.norm(weights, dim=0)

        else:
          # we make the weights orthogonal (true for infinite dim limit)
          weights = torch.eye(H_T)

          # we append on the extra dimensions -- which are all 0
          weights = torch.cat((weights, torch.zeros(dim - H_T, H_T)), 0)


        # instantiate parameters
        self.w = nn.Parameter(weights)
        self.relu = nn.ReLU()
        self.H_T = H_T


    def forward(self, x):
        """
        This is simply a dot product with the weights, then relu. If there are more than one weights, they are averaged
        """
        y = torch.sum(self.relu(torch.einsum('dh,nd->nh', self.w, x)), dim=1) / self.H_T

        return y.view(y.shape[0], 1)



# the learner model

class Perceptron(nn.Module):
    
    def __init__(self, H_L, dim, regime = 'feature'):

        super().__init__()
        
        self.H_L = H_L
        self.regime = regime


        # initialize the network according to the regime we want

        # these are the weights, w in our formula
        self.first_layer = nn.Linear(dim, H_L, bias = False)

        # this is the ReLU and weights a_h in our formula
        self.network = nn.Sequential(
              nn.ReLU(),
              nn.Linear(H_L, 1, bias = False)
          )
        

        if self.regime == 'random_feature':

          # the weights are not updated if we have random feature regime
          self.first_layer.requires_grad_(False)


    def forward(self, x):

        return self.network(self.first_layer(x)) / self.H_L


class Sym_Model:


  def __init__(self, H_L, H_T = 1, dim = 2,  regime = 'feature', seed = None, orthogonal_teacher_weights = False):

    # initialize the neural network according to parameters given by input:
    # H_L - width of learner neural network
    # H_T - Width of teacher NN
    # dim - the dimension of the data/weight vectors
    # regime - either "feature" or "random_feature" for either learning style
    # seed - seeds the random pytorch initialization. Note that training can have a different seed call


    # initialize attributes
    self.H_L = H_L
    self.H_T = H_T
    self.dim = dim
    self.regime = regime

    # error catching
    if regime == 'random feature':
      # correct the user if they typed it wrong
      self.regime = 'random_feature'

    regimes = ['feature', 'random_feature']

    if not(self.regime in regimes):

      raise Exception("regime not recognized, please try 'random feature' or 'feature'")


    # dummy variables (placeholders)
    self.N_train = None
    self.N_test = None
    self.epochs = None


    # seed pytorch
    if seed:
      torch.manual_seed(seed)


    # initialize teacher
    self.teacher = Teacher_sphere(dim, H_T, orthogonal_weights = orthogonal_teacher_weights)
    self.teacher.requires_grad_(False)

    # initialize learner
    self.model = Perceptron(H_L, dim, regime)



  def reinitialize_model(self, seed = None, orthogonal_teacher_weights = False):
    # reininitializes model for a new trial using the exact same parameters
    # can be seeded

    # seed pytorch
    if seed:
      torch.manual_seed(seed)

    # initialize teacher
    self.teacher = Teacher_sphere(self.dim, self.H_T, orthogonal_weights = orthogonal_teacher_weights)
    self.teacher.requires_grad_(False)

    # initialize learner
    self.model = Perceptron(self.H_L, self.dim, self.regime)



  def train_model(self, N_train, N_test, 
  data_spacing = 100, epochs = 100000, lr=50, verbose = False, seed = None, l2_reg = None):
    # N_train - the number of randomly generated training points to train with
    # N_test - the number of points with which test error is calculated
    # data_spacing - the frequency at which test error is calculated and appended to data
    # epochs - the number of iterations of GD
    # lr - the learning rate (keep it \propto H_L, small (0.3*H_L) for FL, large for RFL (5*H_L))
    # verbose - prints the train/test error 10 times during training
    # seed - the pytorch seed with which the data is generated
    # l2_reg - the lambda factor for l-2 regularization


    # we perform a training loop on our model, using MSE and full gradient descent


    # we need this information to name the data file when we save it
    self.N_train = N_train
    self.N_test = N_test
    self.epochs = epochs

    # seed pytorch
    if seed:
      torch.manual_seed(seed)
    
    # generate the data for test and train
    X_tr = torch.randn(N_train, self.dim)
    X_tr /= torch.norm(X_tr, dim = 0)
    X_te = torch.randn(N_test, self.dim)
    X_te /= torch.norm(X_te, dim=0)
    y_tr = self.teacher(X_tr)
    y_te = self.teacher(X_te)

    # put them all on the GPU
    X_tr = X_tr.to('cuda')
    X_te = X_te.to('cuda')
    y_tr = y_tr.to('cuda')
    y_te = y_te.to('cuda')
    self.model = self.model.to('cuda')

    # initialize data 
    data = {
        'test error': [],
        'train error': [],
        'steps': [],
        'weights': [],
        'teacher weights': []
    }


    # loss parameters
    loss = nn.MSELoss()

    # one optimizer with l2 regulrization, one without
    if l2_reg:
      print('using l2 reg')
      optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay = l2_reg)
    else:
      optimizer = optim.SGD(self.model.parameters(), lr=lr)



    # training
    for epoch in range(epochs):

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      y_pred = self.model(X_tr)
      l = loss(y_pred, y_tr)
      l.backward()

      optimizer.step()


      # calculate test loss every 'data_spacing' number of steps, append all data
      if epoch % data_spacing == 0:

        with torch.no_grad():
        
          y_pred_te = self.model(X_te)
          test_loss = torch.mean( (y_pred_te.squeeze() - y_te.squeeze())**2 )

        # append the data
        data['test error'].append(test_loss.item())
        data['train error'].append(l.item())
        data['steps'].append(epoch + 1)


      # debugging
      if verbose and epoch % round(epochs / 10) == 0:

        print('epoch %d\ntrain mse: %E\ntest mse: %E' % (epoch, l, test_loss.item() ))
        #print(y_pred[:5].T, y_tr[:5].T)

    # put the model back on the cpu and take the final distribtion of weights
    self.model = self.model.to('cpu')
    data['weights'].append([copy.deepcopy(w).cpu().detach().numpy() for w in self.model.parameters()])
    data['teacher weights'].append([copy.deepcopy(w).cpu().detach().numpy() for w in self.teacher.parameters()])
    self.data = data


    return 1


  def save(self, path='./', naming = 'auto', reduced_size = False):
    # saves the data in a serialized .npy file
    # path - folder in which it is saved
    # naming - the name of the file. The default option is "<N_training>_<dim>_<epochs>_<regime>_N.npy" 
    # where N means the Nth file of this name
    # final_test_error_only - to save space, we save only the final value of test error to a file

    full_path = self.naming_scheme(path, naming, 'npy')

    data = copy.copy(self.data)

    if reduced_size:

      # decrease file size by a huge margin (only include weight distribution and final test error)
      data.pop('train error')
      data.pop('steps')
      data['test error'] = data['test error'][-1]


    np.save(full_path, data)

    return 1


  def plot_training_curve(self, title = None, save = False, path='./', naming = 'auto', ext = 'pdf'):
      # title - title of the plot
      # save - whether to save or not
      # path - the file in which to save the figure
      # naming - the name to save the figure as, auto will automatically name based on params
      # ext - the extension of the figure, either png or pdf are typical

    # plots and/or saves the training curve, both training mse and test mse over steps
    plt.figure(figsize=[8,4], dpi=90)

    # custom title
    if title:
        plt.title(title)
    else:
        plt.title('MSE Training Loss and MSE Test Error', fontsize = 12)

    plt.xlabel('epochs', fontsize = 12)
    plt.ylabel('MSE', fontsize = 12)
    plt.grid()
    #plt.ylim([0, np.max([np.max(self.data['test error']), np.max(self.data['train error'])])])
    plt.loglog(self.data['steps'], self.data['test error'], color='black', linestyle = '--', marker = '', label = 'Test MSE')
    plt.loglog(self.data['steps'], self.data['train error'], color='red', linestyle = '-', marker = '', label = 'Training Loss (MSE)')
    plt.legend(fontsize=12)


    if save:

        full_path = self.naming_scheme(path, naming, ext)
        plt.savefig(full_path)

    else:
        plt.show()

    plt.close()

    return 1


  def plot_weight_angle_distribution(self, angular_bins = 200):

    ########## WORKS ONLY WITH dim=2 and H_T=1 ############
    
    # we plot the angles of the weights to see if they align with the teacher
    print(self.N_train)

    # calculate the angles of the final weight configuration
    # weights1 is the w in our formula and a is weights2
    weights1, weights2 = self.data['weights'][0]


    # WEIGHT ANGULAR DISTRIBUTION
    
    # we arbitrarily choose w[0] to be the x-axis and w[1] to be the y-axis
    w_angles = np.arctan2(weights1[:,1], weights1[:,0])


    # the true angle of the teach weight
    w_true = [copy.copy(w).detach().numpy() for w in self.teacher.parameters()][0]
    w_angle_true = np.arctan2(w_true[1], w_true[0])
    a = weights2[0]
    a_mags = np.abs(weights2[0])

    # plot a histogram of angles, and plot the true angle
    # plots and/or saves the training curve, both training mse and test mse over steps
    plt.figure(figsize=[8,4], dpi=90)
    plt.grid()
    plt.title('2-Dimensional Learner Angular Weight Distribution Times |a| and Teacher True Weight Angle', fontsize = 12)
    plt.xlabel(r'$\theta$ (rad)', fontsize = 12)
    plt.ylabel('Frequency', fontsize = 12)
    n, _, _ = plt.hist(w_angles, bins = angular_bins, weights= a_mags)
    plt.vlines(w_angle_true, 0, np.max(n), color = 'black', label='True angle', alpha = 0.5, linestyle = '--')
    plt.legend(fontsize=12)
    plt.show()


    # WEIGHT MAGNITUDE DISTRIBUTIONS

    # we plot the distribution of the magnitude of weights w
    w_mags = np.linalg.norm(weights1, axis=1)

    plt.figure(figsize=[8,4], dpi=90)
    plt.grid()
    plt.title(r'2-Dimensional Learner Weight Mangitude Distribution for $\vec{w}_h$', fontsize = 12)
    plt.xlabel('Magnitude', fontsize = 12)
    plt.ylabel('Frequency', fontsize = 12)
    n, _, _ = plt.hist(w_mags)
    
    plt.show()

    # we plot the distribution of the magnitude of weights a
    

    plt.figure(figsize=[8,4], dpi=90)
    plt.grid()
    plt.title(r'Weight Distribution for $a_h$', fontsize = 12)
    plt.xlabel('Magnitude', fontsize = 12)
    plt.ylabel('Frequency', fontsize = 12)
    n, _, _ = plt.hist(a)
    plt.show()


    # SCATTER PLOT IN WEIGHT SPACE

    # Scale w according to their corresponding a, since larger a means it contributes more to the prediction
    scaled_weights = weights1 
    scaled_weights[:,0] *= a_mags
    scaled_weights[:,1] *= a_mags

    max_mag = np.max(np.linalg.norm(scaled_weights, axis=1)) + 0.5

    plt.figure(figsize=[8,8], dpi=90)
    plt.grid()
    plt.xlim([-max_mag, max_mag])
    plt.ylim([-max_mag, max_mag])
    plt.hlines(0, -max_mag, max_mag)
    plt.vlines(0, -max_mag, max_mag)
    
    plt.title(r'Weights in Weight Space Multiplied by Their Corresponding $|a|$', fontsize = 12)
    plt.scatter(scaled_weights[:,0], scaled_weights[:,1], marker = '.', color= 'black', alpha = 0.5, label='Student Weight')
    plt.scatter(w_true[0], w_true[1], marker = 'o', color = 'red', label = 'Teacher Weight')
    plt.xlabel(r'$w_0$', fontsize = 12)
    plt.ylabel(r'$w_1$', fontsize = 12)
    plt.legend()
    plt.show()



  def naming_scheme(self, path, naming, ext):
    # handles naming for several parts where we save stuff

    if naming == 'auto':
        naming = '%d_%d_%d_%s_0.%s' % (self.N_train, self.dim, self.epochs, self.regime, ext)

    naming = naming.replace(" ", "_")

    # make folder if it doesnt exist
    if not(os.path.exists(path)):
        os.makedirs(path)

    full_path = os.path.join(path, naming)

    # if this file name already exists, instead of overwriting it, we add a number to the end
    # the way we handle double digit numbers is weird but works if you have less than 100 instances
    i = 1

    while os.path.exists(full_path):
        full_path = os.path.splitext(full_path)[0]
        
        if i <= 10:
            full_path = full_path[:-1] + str(i)
            full_path += '.' + ext

        else:
            full_path = full_path[:-2] + str(i)
            full_path += '.' + ext

        i += 1

    return full_path




  def get_teacher_weight_fits(self, tol = 5e-2):

    # to test if the student has fitted a teacher in the infinite dimension limit, we have to see if it evaluates
    # f(w^*) correctly. It should evaluate to 1/H_T, thus we can test if this is true.
    # returns array of 0s and 1s -- for not fitted and fitted weights


    teacher_weights = [copy.copy(w).detach() for w in self.teacher.parameters()][0].T

    fitted = np.zeros(len(teacher_weights))

    for i in range(teacher_weights.shape[0]):

      w = teacher_weights[i,:]

      if torch.abs(self.H_T * self.model(w) - 1) < tol:
        fitted[i] += 1


    return fitted



