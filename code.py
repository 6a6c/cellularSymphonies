import numpy as np
import midiutil as mu
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import time

import params

MIN_VEL = 20
MAX_VEL = 120
NUM_VELS = 10
counts = [[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]]
wh = [0, 0, 0]


def major(base, offset):
    scale = [0, 2, 4, 5, 7, 9, 11, 12]
    return int(base + scale[offset % 8] + (12 * int(offset / 8)))

def minor(base, offset):
    scale = [0, 2, 3, 5, 7, 8, 10, 12]
    return int(base + scale[offset % 8] + (12 * int(offset / 8)))

def lydian(base, offset):
    scale = [0, 2, 4, 6, 7, 9, 11, 12]
    return int(base + scale[offset % 8] + (12 * int(offset / 8)))

def mixo(base, offset):
    scale = [0, 2, 4, 5, 7, 9, 10, 12]
    return int(base + scale[offset % 8] + (12 * int(offset / 8)))

def blues(base, offset):
    scale = [0, 3, 5, 6, 7, 10, 11, 12]
    return int(base + scale[offset % 8] + (12 * int((offset / 8))))

def chord_tones(affect, base, offset):
    tones = [ [0, 4, 7], # major
              [0, 3, 7], # minor
              [0, 4, 8], # aug
              [0, 3, 6]] # dim
    return int(base + tones[affect][offset % 3] + (12 * int((offset /  8))))

def get_note(affect, key, instr_base, off):
                # chord | major | minor | lydian | mixo | blues 
    choice = params.note_probs 
    r =  np.random.rand()
    
    if( r <= choice[affect][0] ):
        counts[affect][0]  += 1
        
        c = chord_tones(affect, key + instr_base, off)
        #print("aff of ", affect, ", key of ", key, " and off of ", off, " gave ", c)
        return c

    elif( r <= choice[affect][1] ):
        counts[affect][1] +=  1
        return major(key + instr_base, off)
    elif( r <= choice[affect][2] ):
        counts[affect][2] +=1
        return minor(key + instr_base, off)
    elif( r <= choice[affect][3] ):
        counts[affect][3] += 1
        return lydian(key + instr_base, off)
    elif( r <= choice[affect][4] ):
        counts[affect][4] += 1
        return mixo(key + instr_base, off)
    elif( r <= choice[affect][5] ): 
        counts[affect][5] +=  1
        return blues(key + instr_base, off)
    else: return -1

def get_vel(value):
    return int(MIN_VEL + ((MAX_VEL - MIN_VEL) / NUM_VELS) * value)

def WriteNote(time, div, vels, offs, affect, key, instr_base, track, file, wait_passed):
    wait = wait_passed -  960
    if(wait > 959): return wait

    act_div = [0, 0, 3838, 1918, 958, 0, 478, 318, 958, 238]
    num  = [0, 0,  1, 1, 1, 0, 2, 3, 1,  4]
    if(act_div[div] == 0): return 960
    if(act_div[div] == 3858 and time % 3840 != 0): return wait
    if(act_div[div] == 1918 and time % 1920 != 0): return wait

    if(div == 2 or div == 3): wh[div-1] += 1

    for i in range(num[div]):
        pitch =  get_note(affect, key, instr_base, int(offs[i]))
        vel =  get_vel(int(vels[i]))    
        file.addNote(track, 0, pitch, time + ((act_div[div] + 2) * i), act_div[div] - 108, vel) 
    
    if(act_div[div] == 3858): return 3840
    if(act_div[div] == 1918): return 1920
    return 960

def change_chord(key, affect):
    probs = params.chord_probs
    key_to =[ [-1, 5,  7,  9,   8,  4,  5,   7,   2,   4,   0,   1, 9],
              [-1, 0,  4,  5,  7, 8],   
              [-1, 9],
              [-1, 0,  1, 4,  8] ] #dim
        
    aff_to =[ [-1, 0,  0,  1,   0,  1,  1,   1,   1,   0,   2,   3, 0],
              [-1, 3,  0,  1,  1, 0],
              [-1, 1],
              [-1, 1,  0, 1, 3] ]

    r = np.random.rand()

    p_old  =  -1
    i = 0
    for p in probs[affect]:
        if( r < p):
            return (key + key_to[affect][i]) % 12 , aff_to[affect][i]
        
        i += 1

class CA():
    def __init__(self, num_states, exp_num, seed):
        np.random.seed(seed)
        self.seed = seed
        self.num_states = num_states
        self.radius = 1
        self.neighborhood = 2*self.radius+1
        self.exp_num = exp_num
        self.fn = "Experiment"+ str(exp_num) + ".csv"
        self.exp_dir = "experiment" + str(exp_num)
        self.dim = 18 * 3
        self.timesteps = 512
       
        #state_colors = ["black", "blue", "green", "yellow", "orange", red, purp, cyan, magenta, white]
        self.palette = np.array([[0,0,0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 165, 0],
                                 [232, 17, 35], [104, 33, 122], [0, 188, 242], [236, 0, 140], [255, 255, 255], [17, 17, 148], [18, 231, 100]])

    def simulate(self):
        f = open(self.fn, "w")

        f.write("Wrap:,true\n")
        f.write("K(states):," + str(self.num_states) + "\n")
        f.write("Radius:," + str(self.radius) + "\n")
        f.write("Quiescence:,true\n")    
        f.write("random seed:," + str(self.seed) + "\n")
        
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        # Initialize the board for this experiments
        board_seed = np.random.randint(0, self.num_states, self.dim)
        self.board = np.zeros((self.timesteps, self.dim))
        self.board[0,:] = board_seed
                   
        # Randomly initialize the rule
        to_be_decimated = []
        self.rule_table = [0]*((self.num_states - 1) *3 + 1)
        self.rule_table[0] = 0      # Quiescence 
        sb = "0"
        for x in range(1, (self.num_states - 1)*3 + 1):
            self.rule_table[x] = np.random.randint(1, self.num_states)
            to_be_decimated.append(x)
            sb += str(self.rule_table[x])

        rule_string = sb
        
        # Add in writing to experiment file
        f.write("\n\n")
        f.write("Experiment #:," + str(self.exp_num)+"\n")
        f.write("Rule:," + rule_string+"\n")
        f.write("Step,Entry Zeroed,Class,Lambda,Lambda_t,H,H_t,Fitness,Observations\n")
        
        self.all_boards = []

        index_to_0 = 0
        for z in range(len(self.rule_table)):
            lam_T = self.calculate_lambda_t()
            lam = self.calculate_lambda()
            H_T = self.calculate_H_T()
            H = self.calculate_H()

            if (z == 0):
                entry_zeroed = "-"
            else:
                entry_zeroed = str(index_to_0)

            f.write(str(z) + "," + entry_zeroed + ",," + str(lam) + "," + str(lam_T) + "," + str(H) + "," + str(H_T) + ",,,\n") 
            
            # Randomly select one to be decimated and remove it
            if (len(to_be_decimated) > 0):
                index = np.random.randint(0, len(to_be_decimated))
                index_to_0 = to_be_decimated.pop(index)

            # Step through time updating the board
            for x in range(len(self.board)-1):
                for y in range(len(self.board[x])):
                    self.board[x+1][y] = int(self.rule_table[int(self.calculate_my_sum(x,y))])

            self.all_boards.append(self.board.copy())

            # Create the associated figure 
            fig = plt.figure()
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off() 
            fig.add_axes(ax)
            ax.imshow(self.palette[self.board.astype(int)]) 
            img_fn = self.exp_dir + "/experiment_" + str(self.exp_num) + "_step_" + str(z) + ".png"
            plt.savefig(img_fn, dpi=2048, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # Zero out one of the rule tables
            self.rule_table[index_to_0] = 0 
           
        f.close() 

        return self.all_boards 


    def calculate_lambda_t(self):
        num0 = 0
        for x in range(len(self.rule_table)):
             if (self.rule_table[x] == 0):
                num0 += 1

        return 1.0-(float(num0)/float(len(self.rule_table)))

    def calculate_lambda(self):
        d = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 63, 69, 73, 75, 75, 73, 69, 63, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1]
        num0 = 0
        for x in range(len(self.rule_table)):
            new_state = self.rule_table[x]
            if (new_state == 0):
                num0 += d[x]

        return 1.0-((float(num0)/np.power(self.num_states, self.neighborhood)))

    def calculate_H_T(self):
        state_occurrence = [0]*self.num_states
        for x in range(len(self.rule_table)):
            state_occurrence[self.rule_table[x]] += 1

        H_T = 0
        for x in range(self.num_states):
            ps = float(state_occurrence[x])/float(len(self.rule_table))
            if (ps != 0):
                H_T += (ps*np.log2(ps))

        return -1*H_T

    def calculate_H(self):
        d = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 63, 69, 73, 75, 75, 73, 69, 63, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1]  
        state_occurrence = [0]*self.num_states
        H = 0
        for x in range(len(self.rule_table)):
            new_state = self.rule_table[x]
            state_occurrence[new_state] += d[x]

        for x in range(self.num_states):
            ps = float(state_occurrence[x])/np.power(self.num_states, self.neighborhood)
            if (ps != 0):
                H += ps*np.log2(ps)

        return -1*H

    def calculate_my_sum(self, r, c):
        row = self.board[r]
        s = 0
        for x in reversed(range(1,self.radius+1)):
            index = c-x
            if (index<0):
                s += row[len(row) + index]
            else:
                s += row[index]

        for x in range(0, self.radius+1):
            index = c+x
            if (index >= len(row)):
                s += row[index-len(row)]
            else:
                s += row[index]

        return s

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description="Cellular Automata -- CS 420/527 Project 1")
    
    parser.add_argument("--experiment", "-e", required=True, type=int, help="Experiment number")
    parser.add_argument("--seed", "-s", required=False, type = int, default = -123, help = "Experiment seed")

    args = parser.parse_args()

    instr_names = ["v1", "v2", "va", "vc", "b", "fl", "cl", "ob", "as", "ts", "bs", "fh", "t1", "t2", "tr", "tu", "mb", "pi"]
    instr_progs = [41,     41,   42,   43,  44,   74,   72,   69,   66,   67,   68,   61,   57,  60,    58,   59,   13,   1 ] 
    instr_bases = [72,   60,  48,   36,     24,  72,    60,   48,   60,   48,   36,   48,  60,    60,   36,   24,   48,   60]

    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    aff_names = ["maj", "min", "aug", "dim"]
    
    if(args.seed == -123):
        seed = int(time.time())
    else:
        seed = args.seed

    CA = CA(10, args.experiment, seed)
    all_boards = CA.simulate()

    for step in range(28):
        board = all_boards[step]

        notes = []; vels = []; durs = [];
        for i in range(512):
            notes.append(board[i][0:18])
            vels.append(board[i][18:36])
            durs.append(board[i][36:])

        print(len(notes))

        notes = np.swapaxes(notes, 0, 1)
        vels = np.swapaxes(vels, 0, 1)
        waits = [960] * 18

        myMidi = mu.MIDIFile(18, eventtime_is_ticks = True, deinterleave =  False)
        time = 0
        tempo = 80
        for i in range(18):

            myMidi.addTrackName(i, 0, instr_names[i])
            myMidi.addTempo(i, 0, tempo)
            myMidi.addProgramChange(i, 0, 0, instr_progs[i] -  1)
            

        key = 0
        affect = 0

        for i in range(64):
            if(time % 1920 == 0):
                if(time % 3840 == 0):
                    if( np.random.rand() <= .6):
                        key, affect = change_chord(key, affect)
                        print(time / 4.0, key_names[key] + aff_names[affect], end="   ")
                else:
                    if( np.random.rand() <= .2):
                        key, affect = change_chord(key, affect)
                        print(time / 4.0, key_names[key] + aff_names[affect], end="   ")
            


            start =i*4
            stop = (i+1) * 4
            
            print(waits)

            for j in range(18):
                note_tmp = notes[j][start:stop]
                vel_tmp = vels[j][start:stop]
                dur  =  int(durs[start][j])

                waits[j] = WriteNote(time, dur, vel_tmp, note_tmp, affect, key, instr_bases[j], j, myMidi, waits[j])

            time += 960
                
        print("\n") 
        print(counts)
        print(wh)

        fn = "experiment" + str(args.experiment) + "/output_" + str(step) + ".mid" 
        with open(fn, "wb") as output_file:
            myMidi.writeFile(output_file) 


