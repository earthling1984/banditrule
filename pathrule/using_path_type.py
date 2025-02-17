#
# Copyright 2023 Mithun Vaidhyanathan
# License: GNU License
#
from bandit.core.context import Context
import ast
import io
import bandit
#import hello
#import game_analysis
#import vcg_analysis
from bandit.core import issue
from bandit.core import test_properties as test

import networkx as nx
import pylab
import matplotlib

import pandas as pd 
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linprog

import pandas as pd
import numpy as np
import itertools
import nashpy

import json

r"""
T.B.C
1. global variable to store state that can say whether the path variable was used without the escaping
2. creation of graph
3. game design and analysis
4. modify code base with secure code
"""
str_counter=0
call_counter=0
current_file=""
my_global_graph = nx.DiGraph()#to show the chain of flows - starting from the first file, checking if it's using the sec lib, and then going to the second file, and so on
my_global_graph_for_inout = nx.DiGraph()#to link the files not using sec lib, basically is another/hub-and-spoke view of the global graph
my_global_graph_secure = nx.DiGraph()#to link the files using the sec lib
my_global_graph_undirected = nx.Graph()#to get the average path, as the graph for input validation type issues
desired_global_graph_dir = nx.DiGraph()
desired_global_graph_undir = nx.DiGraph()
files_processed_so_far = []
#Mainly to help with the experiment.
expected_list_of_files = ['']
pass_number = 0

def solve_zerosum_with_linprog(U):
    '''solve_zerosum_with_linprog(): Solve a zero sum game using linear programming
    
        INPUT: U (k*k square matrix), payoffs in zero sum game (opponent gets -U.T)
        OUTPUT: alpha (k-vector) of probability weights for each action (the symmetric equilibrium)
    '''
    k, k2 = U.shape
    assert k == k2, f'Input matrix must be square, got {k}*{k2}'

    oo = np.zeros((1,k))
    ii = np.ones((1,k))

    # objective: c = [-1, 0, 0, ..., 0]
    c = np.insert(oo, 0, -1.0) # insert -1 in front (pos = index 0)
    
    # inequality constraints A*x <= b
    # top = [ 1 ...
    #         1 -1*A.Tl
    #         1  ...  ]
    # bot = [ 0 -1 0 0 
    #         0 0 -1 0 
    #         0 0 0 -1]
    top  = np.hstack( (ii.T, -1*U.T) )
    bot  = np.hstack( (oo.T, -1*np.eye(k)) )
    A_ub = np.vstack((top, bot))
    
    b_ub = np.zeros((1, 2*k))
    b_ub = np.matrix(b_ub)
    
    # contraints Ax = b
    # A = [0, 1, 1, ..., 1]
    A = np.matrix(np.hstack((0, np.ones((k,)))))
    b = 1.0 # just one condition so scalar 

    # v and alpha must be non-negative
    bounds = [(0,None) for i in range(k+1)]

    # call the solver
    sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A, b_eq=b)
    
    # remove the first element: just return the Nash EQ 
    alpha = sol.x[1:]
    return alpha


def best_response(U, i): 
    """best_response(): 
        INPUTS: 
            U: list of payoff matrices 
            i: (int) player for whom to do the best response 

        OUTPUT: 
            br: (NEQ*2) matrix, where br[:,0] is opponent strategies
                and br[:,1] are the best responses. If one strategy a
                has multiple best responses, then there will be several
                columns in br with br[:,0]==a. 
    """
    j = 1-i # opponent
    if i == 0: 
        Ui = U[0]
    elif i == 1: 
        Ui = U[1].T # so that i becomes row player 
    else: 
        raise Exception(f'Not implemented for n>2 players, got i={i}')

    nai, naj = Ui.shape

    # initialie 
    br = []

    for aj in range(naj):
        # column of U corresponding to the aj'th action of the opponent
        Ui_j = Ui[:, aj] 

        # find index values for the rows where Ui_j attains the max
        idim = 0 # there will not be more dimensions in our case 
        br_ij = np.where(Ui_j == Ui_j.max())[idim]

        for b in br_ij: 
            br.append([aj, b])

    return np.array(br)
    

def print_payoffs(U, A, round_decimals=None): 
    '''print_payoffs: Nicely formatted for a 2*2 game 
        INPUTS: 
            U1,U2: (matrices, dim=na1*na2) Payoffs 
            A1: (list of str, len=na1) List of actions of player 1
            A2: (list of str, len=na2) list of actions of player 2
            round_decimals: (int) Number of decimals of precision to print with 
        
        OUTPUT:
            tab: pandas dataframe, na1*na2 with payoff tuples 
    '''
    assert len(U) == 2, f'only implemented for 2-player games'
    assert len(A) == 2, f'only implemented for 2-player games'

    U1 = U[0]
    U2 = U[1]
    A1 = A[0]
    A2 = A[1]

    print(U1)
    print(U2)
    print(A1)
    print(A2)

    na1,na2 = U1.shape
    assert len(A1) == na1
    assert len(A2) == na2

    if not (round_decimals is None):
        assert np.isscalar(round_decimals), f'round_decimals must be an integer' 
        U1 = U1.round(round_decimals)
        U2 = U2.round(round_decimals)

    # "matrix" of tuples 
    X = [[(U1[r,c],U2[r,c]) for c in range(na2)] for r in range(na1)]

    # dataframe version 
    tab = pd.DataFrame(X, columns=A2, index=A1)
    
    return tab 

def find_undominated_actions(U_in, i, A, DOPRINT=False):
    '''find_undominated_actions: finds the actions for player i that are
        not strictly dominated by another action
        
        INPUTS: 
            U_in: (matrix, na1*na2) Payoffs (player 1, player 2)
            i: (integer) Which player we are currently examining
            A: (list) List of actions (len = # of actions for this player)
            
        OUTPUT: 
            AA: (list) undominated actions 
            IA: (list of integers) integers i s.t. AA = [A[i] for i in IA]
            ANYDOMINATED: (bool) True if at least one action was strictly dominated
    '''
    
    AA = []
    IA = []
    nA = len(A)
    
    # 1. ensure that U has actions of player i along 0th dimension 
    if i == 0: 
        # 1.a already the case 
        U = np.copy(U_in)
    else: 
        # 1.b transpose 
        U = U_in.T 
    
    # 2. determine if each action has other dominated actions 
    for ia in range(nA): 
        DOMINATED = False 
                
        for ia_ in range(nA): 
            # 2.a loop through all *other* strategies 
            if ia_ == ia: 
                continue

            # 2.b check if ia_ always gives a higher payoff than ia (i.e. domination)
            if np.all(U[ia_] > U[ia]): 
                DOMINATED = True
                break # exit search: enough that we have found one 
        
        # 2.c append or not 
        if not DOMINATED: 
            AA.append(A[ia])
            IA.append(ia)
            
    # 3. convenient boolean 
    ANYDOMINATED = (len(AA) < len(A))
    
    return AA,IA,ANYDOMINATED


def IESDS(A, U, DOPRINT=False, maxit=10000): 
    '''Iterated Elimination of Strictly Dominated Strategies 
        INPUTS: 
            A: (list of lists) n lists (one for each player), 
                    each has len = # of actions to player i
            U: (list, len=n) list of na1*na2 matrices of payoffs
            DOPRINT: (bool) whether to print output to terminal 
            maxit: (int) break algorithm if this count is ever reached
                (note: the algorithm is not approximate so we can compute 
                what maxit is in the worst case)
        OUTPUT: Actions and payoffs for the undominated game
            A_undominated: (n-list of vectors) 
            U_undominated: (n-list of matrices of payoffs)
    '''
    
    U_undominated = copy.copy(U)
    A_undominated = copy.copy(A)
    
    n = len(U)
    na1,na2 = U[0].shape

    # checks 
    assert n == 2, f'Code only implemented for 2-player games '
    assert len(A) == n
    for i in range(n): 
        assert len(A[i]) == U[i].shape[i]
        assert U[i].shape == (na1,na2), f'Payoff matrix for player {i+1} is {U[i].shape}, but {(na1,na2)} for player 1'

    # initialize flags 
    D = np.ones((n,), dtype='bool')
    
    for it in range(maxit): 

        for i in range(n): 
            # find undominated actions 
            A_undominated[i], IA, D[i] = find_undominated_actions(U_undominated[i], i, A_undominated[i], DOPRINT)

            # if we found at least one, remove it/them from the game 
            if D[i]: 
                # remove from both players' payoff matrices 
                for j in range(n): 
                    if i == 0: 
                        U_undominated[j] = U_undominated[j][IA, :]
                    else: 
                        U_undominated[j] = U_undominated[j][:, IA]


        # break once we have run an iteration without finding any strategies to remove 
        if D.any() == False: 
            break

    return A_undominated, U_undominated

def compute_full_matrix(U1, U2, p, action_names=None):
    """      
    "        Source of this method: https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/BNE.ipynb?short_path=6214124
    "        Assumes that only player 2's type varies \n",
    "        (this means that player 1 has one action per row in U1, \n",
    "         while 2 has nA2**2 (one choice per type))\n",
    "        Both players have one utility matrix for each realization \n",
    "        of player 2's type. \n",
    "         \n",
    "        INPUTS: \n",
    "            U1: list of 2 payoff matrices for player 1 (row player)\n",
    "            U2: list of 2 payoff matrices for player 2 (column player)\n",
    "            p: (scalar) Probability that player 2 is the first type \n",
    "            action_names: [optional] 2-list of names of actions (nA1 and nA2 long)\n",
    "        OUTPUTS: \n",
    "            t1, t2: wide-form payoff matrices suitable for finding the NE \n",
    "            A1, A2: names of actions \n",
    """
    assert len(U1) == 2
    assert len(U2) == 2
    assert np.isscalar(p)
    nA1, nA2 = U1[0].shape
    t1 = np.empty((nA1, nA2*nA2))
    t2 = np.empty((nA1, nA2*nA2))
    
    for ia1 in range(nA1):
        i_col = 0
        
        for a2_1 in range(nA2):
            for a2_2 in range(nA2):
                t1[ia1,i_col] = p*U1[0][ia1,a2_1] + (1.-p)*U1[1][ia1,a2_2]
                t2[ia1,i_col] = p*U2[0][ia1,a2_1] + (1.-p)*U2[1][ia1,a2_2]
                
                i_col +=1
                
    if action_names is None:
        A1 = [f'{i}' for i in range(nA1)]
        A2 = [f'{a}{b}' for a in range(nA2) for b in range(nA2)]
    else:
        assert len(action_names) == 2
        A1 = action_names[0]
        assert len(A1) == nA1, f'Incorrect # of action names'
        a2 = action_names[1]
        assert len(a2) == nA2, f'Incorrect # of action names'
        
        A2 = [f'{a}{b}' for a in a2 for b in a2]
        
    return t1, t2, A1, A2

def print_my_global_graph():
    print("Printing my_global_graph")
    nx.draw(my_global_graph, with_labels=True, font_weight='bold')
    pylab.show()

def print_my_global_graph_for_inout():
    """
    This would be the graph that will help with identifying design issues. Envisioned for situations where a secure library
    is expected to be used, but isn't used in that expected location.
    """
    print("Printing my_global_graph_for_inout")
    nx.draw(my_global_graph_for_inout, with_labels=True, font_weight='bold')
    pylab.show()
    
def print_my_global_graph_secure():
    print("Printing my_global_graph_secure")
    nx.draw(my_global_graph_secure, with_labels=True, font_weight='bold')
    pylab.show()
    
r"""
A function to calculate and return payoff values based on VCG concepts. Takes 2 parameters - a basis for the vulnerability category,
and a structure of the actual code to be analyzed. For example, speaking of the selfish routing example, we figure out in the graph who
the node operators are, and what the cost of a route is in the presence and absence of that operator. That difference becomes 
the payoff for that operator - could either be a negative or a positive payoff. For our case, the developer and the security players 
are the operators. 
"""
def get_payoffs_from_vcg(basis, structure):
    a=0#dummy value to get rid off the IDE error
    
@test.checks("Call")
@test.test_id("B705")
def is_path_there_call(context):
    filenamefullpath = context.filename
    filefullpathtokens = filenamefullpath.split("\\")
    filename=filefullpathtokens[(len(filefullpathtokens)-1)]
    print("================================start writing context object================================")
    print("--------","Files processed so far",":",files_processed_so_far,"--------")
    print("Context node type is",type(context.node))
    global call_counter
    call_counter=call_counter+1
    print("call counter is",call_counter)
    contextrepr = str(context.__repr__())
    print("--------","The context object is of the type",type(context))
    print("--------","The contextrepr object is of the type",type(context.__repr__()))
    print("--------","Context",":",contextrepr,"--------")
    print("--------","Call args",":",context.call_args,"--------")
    print("--------","Call args count",":",context.call_args_count,"--------")
    print("--------","Call function name",":",context.call_function_name,"--------")
    print("--------","Call function qual name",":",context.call_function_name_qual,"--------")
    print("--------","Call keywords",":",context.call_keywords,"--------")
    print("--------","Node",":",context.node,"--------")
    print("--------","Filename",":",context.filename,"--------")
    #print("--------","Line Number",":",context.get_lineno_for_call_arg(),"--------")
    print("--------","String val",":",context.string_val,"--------")
    print("--------","Bytes val",":",context.bytes_val,"--------")
    print("--------","String val as escaped bytes",":",context.string_val_as_escaped_bytes,"--------")
    print("--------","Statement",":",context.statement,"--------")
    print("--------","Function def defaults qual",":",context.function_def_defaults_qual,"--------")
    print("--------","The file data is",context.file_data)
    print("--------","The context file data is of the type",type(context.file_data))
    print("================================end writing context object================================")
    
    r"""
    The Context repr is a tricky representation. It cannot be directly loaded via json.loads, as it's not a standard json.
    The below sections clean the json, with the main goal being to get the line number where the issue is found. 
    """
    first_curly_bracket_open = contextrepr.find("{")
    last_curly_bracket_close = contextrepr.rfind("}")
    
    print("First curly bracket open found at index location",first_curly_bracket_open,"and the last curly bracket close found at index",last_curly_bracket_close)
    
    context_json_str = contextrepr[first_curly_bracket_open:(last_curly_bracket_close+1)]
    context_json_str = context_json_str.replace("'","\"")
    bad_import_curly_bracket_open_approx = 0
    bad_import_curly_bracket_close_approx = 0
    if "imports" in context_json_str:
        bad_import_curly_bracket_open_approx = context_json_str.index("imports")
    if "import_aliases" in context_json_str:
        bad_import_curly_bracket_close_approx = context_json_str.index("import_aliases")
    context_json_str_1=""
    context_json_str_2=""
    context_json_str_3=""
    
    print("Found bad_import_curly_bracket_open_approx at",bad_import_curly_bracket_open_approx)
    print("Found bad_import_curly_bracket_close_approx at",bad_import_curly_bracket_close_approx)
    
    context_json_str_1=context_json_str[:bad_import_curly_bracket_open_approx]
    print("context_json_str_1",context_json_str_1)
    context_json_str_2=context_json_str[bad_import_curly_bracket_open_approx:bad_import_curly_bracket_close_approx]
    print("context_json_str_2 before replace",context_json_str_2)
    context_json_str_2 = context_json_str_2.replace("{","[")
    context_json_str_2 = context_json_str_2.replace("}","]")
    print("context_json_str_2 after replace",context_json_str_2)
    context_json_str_3=context_json_str[bad_import_curly_bracket_close_approx:]
    print("context_json_str_3",context_json_str_3)
    
    context_json_str_prefinal=context_json_str_1+context_json_str_2+context_json_str_3
    context_json_str_prefinal = context_json_str_prefinal.replace("<","\"")
    context_json_str_prefinal = context_json_str_prefinal.replace(">","\"")
    print("--------","context_json_str_prefinal",":",context_json_str_prefinal,"--------")
    
    bad_file_data_dq_open_approx = 0
    bad_file_data_dq_close_approx = 0
    if "file_data" in context_json_str:
        bad_file_data_dq_start_approx = (context_json_str.index("file_data")+len("file_data")+6) #adding 6 at the end to account for standard json chars and skip 5 of them - 2 double quotes around the key, which is followed by a :, the space, and the single quote at the start of the value. We then replace 2 double quotes from that index, as that is the non-standard char for json at this stage.
    context_json_str_4=""
    context_json_str_5=""
    
    context_json_str_4=context_json_str_prefinal[:bad_file_data_dq_start_approx]
    print("context_json_str_4",context_json_str_4)
    context_json_str_5=context_json_str_prefinal[bad_file_data_dq_start_approx:]
    print("context_json_str_5 before replace",context_json_str_5)
    context_json_str_5=context_json_str_5.replace("\"", "'", 2)
    print("context_json_str_5 after replace",context_json_str_5)
    
    context_json_str_final=context_json_str_4+context_json_str_5
    
    context_json = json.loads(context_json_str_final)
    print("--------","context_json",":",context_json,"--------")
    
    target_line_number = (context_json["lineno"]-1)
    
    inloop_file_to_be_loaded = open(context.filename)
    print("Will test opening the file",inloop_file_to_be_loaded)
    for i, line in enumerate(inloop_file_to_be_loaded):
        print("Inside the for loop for line number",i)
        if i == target_line_number:
            print("The issue has been found at line number",target_line_number)
    inloop_file_to_be_loaded.close()
    

    #T.B.C the actual values vcg_analysis.main()
    my_graph = nx.DiGraph() 
     
    # Add edges to to the graph object
    # Each tuple represents an edge between two nodes
    my_graph.add_weighted_edges_from([
                            (1,2,3.0), 
                            (1,3,2.0), 
                            (2,4,2.0), 
                            (3,5,3.0), 
                            (2,5,1.0),
                            (3,6,5.0),
                            (4,6,2.0),
                            (5,6,1.0)])
     
    # Draw the resulting graph
    #nx.draw(my_graph, with_labels=True, font_weight='bold')
    #pylab.show()
    shortest_path = nx.dijkstra_path(my_graph, 1, 6)
    shortest_path_distance = nx.shortest_path_length(my_graph, 1, 6, 'weight', 'dijkstra')
    print("The shortest path between 1 and 6, with EF (56), is",shortest_path)
    print("The distance travelled in the shortest path, with EF (56), is",shortest_path_distance)
    
    my_graph_2 = nx.DiGraph() 
     
    # Add edges to to the graph object
    # Each tuple represents an edge between two nodes
    my_graph_2.add_weighted_edges_from([
                            (1,2,3.0), 
                            (1,3,2.0), 
                            (2,4,2.0), 
                            (3,5,3.0), 
                            (2,5,1.0),
                            (3,6,5.0),
                            (4,6,2.0)])
     
    # Draw the resulting graph
    #nx.draw(my_graph_2, with_labels=True, font_weight='bold')
    #pylab.show()

    if context.call_args:
        for call_arg in context.call_args:
            if call_arg is not None:
                if 'path:' in call_arg:
                    return bandit.Issue(
                        severity=bandit.HIGH,
                        confidence=bandit.HIGH,
                        text="Use of the flask type, path, detected, that can cause XSS. Never accept an entire path from the user. Instead change the data type here to string."
                    )
    
    #checking for an experimental use case
    if (context.call_function_name_qual=='html.escape' and not 'usepath.py' in context.filename):
        print("We want the security logic around html.escape to be in usepath.py, but it was found in", context.filename)
        print("If the outcome of the Game Theoretic analysis is to finx, this plugin will checking a copy of the code with the logic in usepath.py. It can be a candidate for merging with the main branch.")
        r"""
        TODO
        1. Call VCG payment calculator with actual values. Values below are dummy ones. (the num of hops can be = the distance from the source to the end validation location, farther it is, more is the num of hops, making it worse for the dev)
        2. With actual payoffs, call game analysis. Values below are dummy ones.
        3. Based on game analysis results, given the equilibriums, finx or don't finx. I.e., modify the code or not. (how to modify the file, from the AST? Look-up.)
        4. For checking the measure of co-operation, any need to calculate Shapley value?
        """

        # Keeping filename unique for starters. Q to ponder: allow duplicate nodes? For capturing any repetition of code blocks?
       
        #Directed and undirected graph long
        #The way to read the global graph - how many times is usepath's logic to be checked in the target file? That number is appended to usepath.py. We chain each usage, and build a graph to showcase the replication
        usepath_this_iter = ''
        global_graph_num_of_edges=my_global_graph.size()
        print('Size of the vulnerable global graph, i.e. the number of edges, is',global_graph_num_of_edges)
        if global_graph_num_of_edges == 0:
            usepath_this_iter = 'usepath.py (0)'
            my_global_graph.add_node(filename)
            print("adding the node for", filename)
            my_global_graph.add_node(usepath_this_iter, type='usepath')
            print("adding the node for", usepath_this_iter)
            my_global_graph.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
            my_global_graph_undirected.add_node(filename)
            print("adding the node for", filename)
            my_global_graph_undirected.add_node(usepath_this_iter, type='usepath')
            print("adding the node for", usepath_this_iter)
            my_global_graph_undirected.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
        elif global_graph_num_of_edges == 1:
            usepath_last_iter = 'usepath.py ('+str(global_graph_num_of_edges-1)+')'
            usepath_this_iter = 'usepath.py ('+str(global_graph_num_of_edges)+')'
            my_global_graph.add_node(filename)
            print("adding the node for", filename)
            my_global_graph.add_node(usepath_this_iter, type='usepath')
            print("adding the node for", usepath_this_iter)
            my_global_graph.add_weighted_edges_from([(usepath_last_iter,filename,0.0)])
            my_global_graph.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
            my_global_graph_undirected.add_node(filename)
            print("adding the node for", filename)
            my_global_graph_undirected.add_node(usepath_this_iter, type='usepath')
            print("adding the node for", usepath_this_iter)
            my_global_graph_undirected.add_weighted_edges_from([(usepath_last_iter,filename,0.0)])
            my_global_graph_undirected.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
        else:
            usepath_last_iter = 'usepath.py ('+str(global_graph_num_of_edges-2)+')'
            usepath_this_iter = 'usepath.py ('+str(global_graph_num_of_edges)+')'
            my_global_graph.add_node(filename)
            print("adding the node for", filename)
            my_global_graph.add_node(usepath_this_iter, type='usepath')
            print("adding the node for", usepath_this_iter)
            my_global_graph.add_weighted_edges_from([(usepath_last_iter,filename,0.0)])
            my_global_graph.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
            my_global_graph_undirected.add_node(filename)
            print("adding the node for", filename)
            my_global_graph_undirected.add_node(usepath_this_iter, type='usepath')
            print("adding the node for", usepath_this_iter)
            my_global_graph_undirected.add_weighted_edges_from([(usepath_last_iter,filename,0.0)])
            my_global_graph_undirected.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
        
        #Directed graph hub-and-spoke
        my_global_graph_for_inout.add_weighted_edges_from([
                                (filename,'usepath.py',2.0)])
        
        return bandit.Issue(
            severity=bandit.HIGH,
            confidence=bandit.HIGH,
            text="The XSS check is not in the main validator, and is insecure, as it increases the fix and maintenance time for such issues. This pattern will prevent the code from scaling."
        )
    elif (context.call_function_name_qual=='callers.usepath.using_path'):
        usepath_this_iter_2 = ''
        secure_global_graph_num_of_edges=my_global_graph_secure.size()
        print('Size of the vulnerable global graph, i.e. the number of edges, is',secure_global_graph_num_of_edges)
        if secure_global_graph_num_of_edges == 0:
            usepath_this_iter_2 = 'usepath.py (0)'
            my_global_graph_secure.add_node(filename)
            print("adding the node for", filename)
            my_global_graph_secure.add_node(usepath_this_iter_2, type='usepath')
            print("adding the node for", usepath_this_iter_2)
            my_global_graph_secure.add_weighted_edges_from([(filename,usepath_this_iter_2,2.0)])
        elif secure_global_graph_num_of_edges == 1:
            usepath_last_iter_2 = 'usepath.py ('+str(secure_global_graph_num_of_edges-1)+')'
            usepath_this_iter_2 = 'usepath.py ('+str(secure_global_graph_num_of_edges)+')'
            my_global_graph_secure.add_node(filename)
            print("adding the node for", filename)
            my_global_graph_secure.add_node(usepath_this_iter_2, type='usepath')
            print("adding the node for", usepath_this_iter_2)
            my_global_graph_secure.add_weighted_edges_from([(usepath_last_iter_2,filename,0.0)])
            my_global_graph_secure.add_weighted_edges_from([(filename,usepath_this_iter_2,2.0)])
        else:
            usepath_last_iter_2 = 'usepath.py ('+str(secure_global_graph_num_of_edges-2)+')'
            usepath_this_iter_2 = 'usepath.py ('+str(secure_global_graph_num_of_edges)+')'
            my_global_graph_secure.add_node(filename)
            print("adding the node for", filename)
            my_global_graph_secure.add_node(usepath_this_iter_2, type='usepath')
            print("adding the node for", usepath_this_iter_2)
            my_global_graph_secure.add_weighted_edges_from([(usepath_last_iter_2,filename,0.0)])
            my_global_graph_secure.add_weighted_edges_from([(filename,usepath_this_iter_2,2.0)])
    #if context.filename not in files_processed_so_far:
        #print_my_global_graph()
    #The below if condition should be after the above one. Else, if you add the file to the list before, the global graph will never be printed
    if context.filename not in files_processed_so_far:
        files_processed_so_far.append(context.filename)
    num_files_processed = len(files_processed_so_far)
    expected_num_of_files = len(expected_list_of_files)
    print("So far, we have processed",num_files_processed)
    print("We expect to process", expected_num_of_files)
    r"""
    Checking for app.run isn't the most elegant check. The reason for this check is to catch the moment when the plugin has completed
    processing all files. Since the plugin itself runs in a loop, until all files are processed, and passes control back to bandit, the ideal
    location to catch the moment will be somewhere suitable in bandit. The reason for such a check is to send the final graph to the game
    logic, and take a decision to finx or not. If done in the plugin, the game will be played again and again for each context and unfinished
    graph, which we don't want. We want it all to happen, after the plugin has done it's work. Due to the Ph.D. time constraints, and the 
    fact that the focus of my topic isn't SAST/compilers in general, or bandit in general, I may consider enhancing things post my degree. 
    For now, this condition is sufficient, for intercepting the moment when the plugin has done it's bit in processing the files. With this
    "hack", we are now able to ensure that the game is played just once.
    """
    if ((num_files_processed == expected_num_of_files) and (context.call_function_name_qual == 'app.run')):
        print("we have processed all the expected files")
        print_my_global_graph()
        print_my_global_graph_for_inout()
        print_my_global_graph_secure()
        r"""
        TBC Below values are dummy ones. Next is to send the above graph below, and do the game analysis on our actual graph
        The question to answer - given a graph of some size, which points to how complex the math behind the defensive code has become,
        can our game analysis work around it, and suggest simpler designs. For our experiment, we have the usage of the path type in flask,
        and the presence of the defensive code expected in one file. When these expectations get violated, we will lean on our game analysis
        to re-write the code, i.e., finx, if the game solution is above some threshold. This "some threshold" will be the probability you will
        see a few lines below. It will be an exciting contribution to make. Basically, for a Bayesian game, the idea we are trying to create
        is that because of the law of conservation of mass/energy, the total payoffs in our universe, which we deem as a closed system, to be
        constant. So, eventually, we can propose that everything is a constant sum game? Either way, for our base calculation, the probability
        value will derive from sources listed, which will provide the number of times certain vulnerability categories have manifested. We will
        align our example scenarios with the most common categories, input validation mainly expected, and employ a probabilistic approach, 
        which gels well with what Bayesian games need by nature.
        """
        shortest_path_2 = nx.dijkstra_path(my_graph_2, 1, 6)
        shortest_path_distance_2 = nx.shortest_path_length(my_graph_2, 1, 6, 'weight', 'dijkstra')
        print("The shortest path between 1 and 6, without EF (56), is",shortest_path_2)
        print("The distance travelled in the shortest path, without EF (56), is",shortest_path_distance_2)
        target_edge_weight = my_graph.get_edge_data(5, 6)['weight']
        print("Confirming that the weight of the edge EF (56) is",target_edge_weight)
        r"""The below formula, from the VCG lecture 3.3, is an easier, and different way to understand the main
        VCG formula for payment to an agent. Here, basically, we find the cost on others when the edge owner
        plays the game, and subtract it with the cost on others when the edge owner doesn't play the game. This
        difference, which is a measure of the costs on others with or without the target player, is the payment
        to/from the player. Payment is made to the player, if the result is negative, and the player pays, when the
        result is positive - standard VCG convention applies.
        """
        payment_ef = ((-shortest_path_distance_2) - (-shortest_path_distance+target_edge_weight))*(-1)
        print("payment to EF (56) is",payment_ef)
        profit_ef = payment_ef-target_edge_weight
        print("profit to EF (56) is",profit_ef)
        #VCG example end
        
        start_nodes_graph_1 = [n for n, d in my_graph.in_degree() if d == 0]
        end_nodes_graph_1 = [n for n, d in my_graph.out_degree() if d == 0]

        print("Start node(s) in graph 1:", start_nodes_graph_1)
        print("End node(s) in graph 1:", end_nodes_graph_1)
        
        start_nodes_graph_2 = [n for n, d in my_graph_2.in_degree() if d == 0]
        end_nodes_graph_2 = [n for n, d in my_graph_2.out_degree() if d == 0]

        print("Start node(s) in graph 2:", start_nodes_graph_2)
        print("End node(s) in graph 2:", end_nodes_graph_2)
        
        start_nodes_global_graph = [n for n, d in my_global_graph.in_degree() if d == 0]
        end_nodes_global_graph = [n for n, d in my_global_graph.out_degree() if d == 0]

        print("Start node(s) in my global graph:", start_nodes_global_graph)
        print("End node(s) in my global graph:", end_nodes_global_graph)
        
        #simrank_overall = nx.simrank_similarity(my_global_graph, None, None, 0.9, 1000, 0.0001)
        #print("The simrank in my overall graph is",simrank_overall)
        avg_shortest_path_length = nx.average_shortest_path_length(my_global_graph_undirected, None, None)
        print("The average shortest path length in my_global_graph_undirected is",avg_shortest_path_length)
        my_in_degree_iter = my_global_graph_secure.in_degree()
        my_out_degree_iter = my_global_graph_for_inout.out_degree()
        
        total_in = 0
        total_out = 0
        for node_in_degree in my_in_degree_iter:
            total_in = total_in+node_in_degree[1]
            if node_in_degree[1]>0:
                print(node_in_degree[0],"has in degree greater than 0, which is equal to",node_in_degree[1])
            
        for node_out_degree in my_out_degree_iter:
            total_out = total_out+node_out_degree[1]
            if node_out_degree[1]>0:
                print(node_out_degree[0],"has out degree greater than 0, which is equal to",node_out_degree[1])
        
        #TBC: incorporating this into the VCG values, plus the relation between CVE severity and global game
        print("Total in flows",total_in)
        print("Total out flows",total_out)
        
        usepath_nodes = []
        non_usepath_nodes = []
        #The custom attribute 'type' needs to be retrieved, as that is what we have defined to keep track of whether a file is using the required security library (usepath) or no. If yes, then the type of the node will be usepath, else no.
        for (p, d) in my_global_graph.nodes(data="type"):
            if d == 'usepath':
                usepath_nodes.append(p)
            else:
                non_usepath_nodes.append(p)
        num_usepath_not_used = len(non_usepath_nodes)
        num_usepath_repetitions = len(usepath_nodes)
        print("The number of files using usepath is",total_in)
        print("The number of files not using usepath, i.e., the repetition of the same logic in usepath, is",num_usepath_not_used)
        print("Usepath repetitions = ",num_usepath_repetitions)
        for node in non_usepath_nodes:
            print(node, "is not calling usepath.py, but is simply replicating the logic. Remove it from here and replace with a call to usepath.py.")
        
        #T.B.C
        #payment = VCG_value*factor_from_conservation_law (VCG value will be in v out metric, or avg path. Compatibility factor will be categoryRglobal)
        #payment = payment*future_x_factor_if_varying_universe (incompatibility factor?)
        
        r"""
        Logic behind the different measurements:
        1. in v. out metrics - for the design issues
        2. Avg. path - for XSS issues+design issues (actually for all categories?)
        3. Relations based metric(which?) - for SQLi. The or 1=1 is a flaw in the way the propositional logic is evaluated. 
        Irrespective of the fix techniques elsewhere, the core issue of the logic abuse has never been fixed. Measuring this can be a good starting point.
        Cyclical graph for tautology? Modeling relations as graphs, and analyzing the properties?
        
        Avg path calculation could be the low level metric, and cat. specific metric the high level. So, avg path could apply to call cats.
        For XSS, the high level metric could be language level, where 3 is good (regex/strict). For SQL, relation metric (TBC)
        
        The avg path is to prevent repetition of the high level defensive measure.
        
        f(avg path)_supergame = sigma_cat f(avg path length)_cat
        OR
        f(metric)_supergame = sigma_cat f(metric)_cat
        OR
        f(metric)_universalgame = sigma_cat f(metric)_cat
        
        in our case
        f(metric)_universalgame = f(in-out-metric)_design_issue + f(avg_path_length)_input_validation + f(tautology_outcome)_SQLi_tautology + f(yet_to_be_realized_games_payoffs)
        tautology metrics - consistency? equiconsistency? soundness? validity?
        
        f(whatever) = operations on matrices? Representation of matrices?
        
        Another thing to do: Make the Spaniel-utility above for n matrices instead of the 2 shown above.
        
        Argument to make - treat security overall (for all categories) as an isolated system, and if any leak, play the game to handle it within
        the other isolated system we are in? or assume from start that security isn't isolated, and compare with functionality etc. The
        "leaking" quantity here is STEM related (let's say gap in logic of one vuln. that doesn't fit into some functionality, and 
        not money or other economic ones. So, each time we assume isolated system, but that isn't the case, we add a matrix. This 
        might be a bit problematic on the time complexity, so would be good to comment on it.
        
        """
        
        math_compatibility_factor = 1 #between 0 and 1?
        math_incompatibility_factor = 1 #between 0 and 1?
        sdlc_cycles = 1 #T.B.C: number of branches/commits can indicate cyclicity factor?
        payment_to_others = (num_usepath_not_used+avg_shortest_path_length)/2
        payment_to_others = payment_to_others*math_compatibility_factor*math_incompatibility_factor
        payment_to_sec = total_in
        payment_to_sec = payment_to_sec*math_compatibility_factor*math_incompatibility_factor
        
        print("Payment to others", payment_to_others)
        print("Payment to sec",payment_to_sec)
        
        # T.B.C the actual values game_analysis.main()
        p = 0.2#To be defined by calling game_helper.get_bayesian_game_probability, from standard datasets provided by authorities+vendors
        #For the probability, we will use the idea of conservation of wealth via material/energy types of wealth
        #Think of this probability as the "guide" on the risk/reward of vuln. categories.
        #T.B.C The math proof for the conservation idea
        #Play around with the probability values and provide various Game Outputs?! Play around for combined Bayesian game, and separate NF games?
        #Where is the conservation/symmetry/compatibility idea most relevant? VCG calculation? Probability calculation? Conservation for risk management and intangible representation, symmetry for design/IV
        #The story to be told here is the merit of combining across categories, and reasoning out which ones to finx/which not (i.e. the dev to fix). If added to defect backlog, then increase penalty.
        r"""
        ---------------Meeting 30-08-2023---------------
    
            Get the probabilities
                from law of conservation? The idea of risk management can be quantified using this approach.
                    where risk = math_compatibility_factor*math_incompatibility_factor?
                Nash Bargaining Solution?
                Size of automata/other automata metrics for input validation?
                    Any other STEM metric that is equivalent to input validation or other vuln categories?
                https://www.cvedetails.com/vulnerabilities-by-types.php
                    total-all-cats-10-years=83557
                    total-sqli-10-years=6877 (0.0823)
                    total-iv-10-years=8096 (0.0969)
                    total-xss=20982 (0.2511)
                Get the dollar value per category from OWASP (and any other source too)
                CWE link?
                https://blog.cloudflare.com/application-security-report-q2-2023/
                https://www.indusface.com/research-reports/state-of-appsec-report-q1-2023.pdf?utm_medium=email&_hsmi=258835016&_hsenc=p2ANqtz--XgW29dM0wuwo-_PwVPzLdjZHDgNuP0uHB1re6MQwOd1tGhEldWS-ZSX5rhhmL9jIGLsxGXYw8q-rYL5hLlUkmdkW6fepoXb1_Nr_pDZ7N8Yj10_U&utm_content=258835016&utm_source=hs_automation
                https://www.datadoghq.com/state-of-application-security/
            Get also a threshold based on the above/new criteria
                CVE/CVSS to above?
                Industry specific? Low v high risk appetite for military (life/physical asset loss) v social media (non-physical loss)?
            Compute the automata metrics for input validation
            
            3/10/2023
            FOR THE CODE
                probab values above
                Get graph/data structure from SAST tool
                    VCG
                    pybandit
                    ??
                    This will give us the payoff matrix, which will solve the game
                    is this a graph v automata, or graph to automata type game to check out?
                """
        
        """
        Example from Spaniel:
        PD: 0.2
                L|R
        Up --> 3,3|0,4
        Down --> 2,1|1,2
        
        XSS/Input Validation/Design Issue (multiply by line numbers, and release cycle count pending)
        Regular Payoff Matrix                            Irregular Payoff Matrix (idea derived from Irregular Matrix/Jagged Arrays)
                Finx/Don't                                Finx/Don't
        Fix    -23,20/-20,-20        OR        Fix        -23,20/-20,-20
                                                
        Don't  0,20/-20+x,-20+x                Don't        0,20/
                
        SH: 0.8
                L|R
        Up --> 3,3|0,2
        Down --> 2,0|1,1
        
        SQL Injection Tautology based (multiply, for a DB vendor, by every user in the world who has to parameterize the query)
        Regular Payoff Matrix                            Irregular Payoff Matrix
                Finx/Don't                                Finx/Don't
        Fix     -23,3/-3,-3                      Fix      -23,3/-3,-3
        
        Don't   0,3/-3+x,-3+x                   Don't     0,3/
        
        The combined matrix (we could argue to re-composing matrices to the universal game here):
                        LL|LR|RL|RR
        Up -->        3,3|0.6,2.2|2.4,3.2|0,2.4
        Down -->    2,0.2|1.2,1|1.8,0.4|1,1.2

        Converted to our game:
        Design/XSS: 0.x - probabilities based on CVE data
                Finx|Don't
        Fix    a,b|c,d
        Don't  e,f|g,h
        a to h values = VCG(returns in-out metrics/avg. path length) based on graph outputs of the analyzed pathrule code
        SQLi: 0.y - probabilities based on CVE data
                Finx/Don't
        Fix    i,j|k,l
        Don't  m,n|o,p
        i to p values = VCG(returns database-graph property such as cyclic graph due to tautology) based on graph outputs of the analyzed db code
        Simple rule - if cycle found, then tautology, hence finx. No questions asked for this category, as it's the DB vendor code (like mysql) 
        Can be less strict for above.
        
        """
        #T.B.C - add SDLC metrics
        #in addition to the graph with in-out metrics, is another graph possible to visualize for the same category with finx. Should those 2 be compared with those of other categories in the Bayesian analysis? Or do we keep them separate as analyses? 
        u1 = np.array([[-23,-23], [0,0]]) #To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
        U1 = [u1, u1]
        A1 = ['U', 'D']
        
        u21 = np.array([[20,-20], [20,-20]]) #To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
        u22 = np.array([[3,-3], [3,-3]]) #randomly made by me, need refining, but more to capture that sec may not finx all - i.e 20 instead of 21 hops, maybe due to false positive/other reason#To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
        U2 = [u21, u22]
        a2 = ['L', 'R']
        A2 = [f'{a}{b}' for a in a2 for b in a2]
        
        print(f'---- If P2 is type 0, the payoffs are -----')
        tab1 = print_payoffs([u1, u21], [A1, a2])
        print(tab1)
        
        print(f'---- If P2 is type 1, the payoffs are -----')
        tab2 = print_payoffs([u1, u22], [A1, a2])
        print(tab2)
        
        print(f'----- Compute full matrix of all types -------')
        t1, t2, A1, A2 = compute_full_matrix(U1, U2, p, [A1, a2])
        tab_combined = print_payoffs([t1, t2], [A1, A2], 3)
        print(tab_combined)
        
        print(f'----- A_ --------')
        A_, T_ = IESDS([A1, A2], [t1, t2], DOPRINT=True)
        strategy_security = A_[1]
        print(type(strategy_security))
        print(A_[1])
        print(f'----- T_ --------')
        print(T_)
        print(f'----- IESDS --------')
        tab_iesds = print_payoffs(T_, A_, 3)
        print(tab_iesds)
        
        eqs = list(nashpy.Game(T_[0], T_[1]).support_enumeration())
        print(f'Found {len(eqs)} equilibria')
        for i,eq in enumerate(eqs):
            print(f'{i+1}: s1 = {eq[0]}, s2 = {eq[1]}')
        for strategy_string in strategy_security:
            for strategy in strategy_string:
                if strategy == 'L':
                    #Need to store some flag for the game type, and finxing the file and location accordingly
                    #Use modularity/cyclicity for decision?
                    print("will finx this file/line")
        #FINAL NUMBERS TO SHOW/INTERPRET - speed/accuracy/something else? - work with experimental values
        #SHOW THEORETICAL STRENGTH FOR RECOMPOSING TO UNIVERSAL GAME? - work with dummy values
        #ARGUE ABOUT MIX OF (THEORETICAL STRENGTH+EXPERIMENTAL STRENGTH)/2?
        #T.B.C if some condition as per above is met (derived from conservation principle and the probability above), then re-write the source, and check-in
        #T.B.C, if A_ for security is L, then finx
        # Unparse e.g.: https://stackoverflow.com/questions/3774162/given-an-ast-is-there-a-working-library-for-getting-the-source
        # Modify AST and write back source: https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c
        #T.B.C If not re-write source, then aspects?
        #repetition/secure code in wrong location is decomposed game, we then combine/re-compose it
        #for file in non_usepath_nodes:
            #print(bandit.linecache.getlines(file,module_globals=None))
    else:
        print("we have not processed all the expected files")