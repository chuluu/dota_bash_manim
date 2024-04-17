# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:16:41 2024

@author: mbluu
"""
from manim import *
import math
import numpy as np
import random

CONFIG = {"include_numbers": True,
          "include_tip": False}
template = TexTemplate(preamble=r'\usepackage{soul}')



def Develop_Img(name,DIR):
    img = ImageMobject(name)
    img.scale(0.4)
    img.to_corner(DIR)
    
    return img

def C_product(C,N):
    product = 1
    for ii in range(N-1):
        product = product*( (1/(ii+1)) - C)
    
    return product

def pseudorand(target):
    C = 0.30210
    prob = 0.5
    
    while abs(target-prob) > 10**-8:
        if C < 0.1:
            C = C + (target-prob)/8
        else:
            C = C + (target-prob)/2
        mat_size = int(np.ceil(1/C))
        P = np.zeros((mat_size,mat_size))
        
        for n in range(P.shape[0]):
            if (n+1)*C < 1:
                P[n,0] = (n+1)*C 
            else:
                P[n,0] = 1
        
        for n in range(P.shape[0] - 1):
            P[n,n+1] = 1- P[n,0]
        
        [v,d] = np.linalg.eig(P.T)
        stationary = d[:,0]/sum(d[:,0])
        prob = np.real(np.dot(stationary,P[:,0]))
        
    return C, prob

def emphasis_box(term,color=YELLOW):
    box = SurroundingRectangle(term, corner_radius=0.2,color=color)
    emphasized_term = VGroup(term,box)
    
    return emphasized_term


class Pseudo_Random(Scene):
    def geometric_pmd(self,p):
        N_tot = 25
        N = [ii+1 for ii in range(N_tot)]
        P = np.zeros(N_tot)
        for n in range(N_tot):
            P[n] = (1-p)**(n) * p
            if P[n] < 0:
                break
        return N, list(P)
        
    def prd_pmd(self, p):
        [C,prob] = pseudorand(p)

        N_tot = 25
        N = [ii+1 for ii in range(N_tot)]
        P = np.zeros(N_tot)

        for n in N:
            P[n-1] = math.factorial(n) * C * C_product(C,n)
            if P[n-1] < 0:
                break
        
        return C, N, list(abs(P))
        
        
    def prd_distribution(self):
        for n in range(1000):
            P_linear.append(C*n)
            N.append(n)
            if P_linear[n] > 1:
                break
            
        return N, P_linear
    
    
    
    def basher_setup_scene(self, player=True):
        img1 = Develop_Img('basher.jpg',UP)
        img1.scale(2)
        box1 = SurroundingRectangle(img1, corner_radius=0.2)
        group1 = Group(img1,box1)
        
        img2 = Develop_Img('bash_description.png',UP)
        img2.scale(4)
        img2.next_to(img1,DOWN)
        
        txt = Text('1 in every 4 hits?').next_to(img2,DOWN)
        
        if player == True:
            self.play(FadeIn(img1))
            self.wait(2)
            self.play(FadeIn(img2))
            self.wait(2)
            self.play(FadeIn(txt))
            self.wait(2)
        else:
            self.add(img1,img2,txt)
        return img1, img2, txt
    
    def first_number_scene(self,player=True):
        x_len = ValueTracker(1)
        axes = always_redraw(lambda: Axes(
                x_range = [0,x_len.get_value(),1],
                x_length = x_len.get_value()*2,
                y_range = [0,1,0.25],
                y_length = 3,
                axis_config=CONFIG,
                ).to_edge(DL).set_color(GREY)
            )
        labels = axes.get_axis_labels(
            Tex("").scale(0.7), Text("P(n)").scale(0.45)
        )
        
        func = always_redraw(lambda: axes.plot(lambda x: 0.25, color = YELLOW))
        Dot_func = always_redraw(lambda: Dot(color = WHITE).scale(1.2).move_to(axes.c2p(1,func.underlying_function(1))))
        
        on_screen_var = Variable(x_len.get_value(), Text("n"), num_decimal_places=0).shift(LEFT)
        var_tracker = on_screen_var.tracker


        plotter_func = VGroup(on_screen_var,axes,func,Dot_func,labels)

        if player == True:
            self.play(Write(on_screen_var))
            self.play(FadeIn(axes),Write(labels))
            self.play(FadeIn(func))
            self.play(FadeIn(Dot_func))
    
            for ii in range(5):
                self.play(var_tracker.animate.increment_value(1),run_time=1)
                self.play(x_len.animate.increment_value(1),run_time=1)
                self.wait(0.5)
            
            self.wait(2)
        else:
            self.add(plotter_func)
        
        
        
        return plotter_func
    
    def uniform_chart(self,plotter_func,player = True):
        N_geo, P_geo = self.geometric_pmd(0.25)
        N_len = len(N_geo)
        chart_x_len = 10
        chart_y_len = 3
        
        box_height = ValueTracker((0.25/0.3) * 3)
        y_midpt = ValueTracker((0.25/0.3) * 0.15)
        x_midpt = ValueTracker(0.5)
        
        chart = BarChart(
            values= P_geo,
            bar_names=N_geo,
            y_range=[0, 0.3,0.1],
            y_length=chart_y_len,
            x_length=chart_x_len,
            x_axis_config={"font_size": 20},
        ).to_edge(DOWN)
        
        labels = chart.get_axis_labels(
            Tex("n").scale(0.7), Text("P(N=n)").scale(0.45)
        )
        
        uni_text_st = Tex(r"\st{Uniform Probability}", tex_template=template).next_to(chart,UP).set_color(RED)
        uni_text = Tex(r"Uniform Probability", tex_template=template).next_to(chart,UP)


        rect2 = always_redraw(lambda: Rectangle(width=chart_x_len/N_len, 
                                                height=box_height.get_value()).move_to(chart.c2p(x_midpt.get_value(),
                                                                                                 y_midpt.get_value())).set_color(YELLOW))
        Scene_tot = VGroup(chart,uni_text_st,labels,rect2)
        if player == True:
            self.wait(2)
            self.play(ReplacementTransform(plotter_func,chart),
                      Write(labels),run_time=1.5)
            self.wait(4)
            self.play(Write(rect2))
            self.wait(3)
            self.play(box_height.animate.set_value((P_geo[3]/0.3) * chart_y_len),
                      y_midpt.animate.set_value((P_geo[3]/0.3) * 0.15),
                      x_midpt.animate.set_value(3.5),run_time=2)
            self.wait(3)
            self.play(FadeIn(uni_text))
            self.wait(2)
            self.play(TransformMatchingTex(uni_text,uni_text_st))
            self.wait(2)
        else:
            self.add(Scene_tot)
        
        return Scene_tot
    
    def prd_Scene1(self,C):
        Talking_Text= Paragraph("We want to hit basher",
                                "1 in every 4 hit",
                                "not 25% per hit.")
        Text_Pseudo = Text('Pseudo Random Distribution')
        Talking_Text.generate_target()
        Talking_Text.target.shift(3*UP)
        
        Pseudo_tot = emphasis_box(Text_Pseudo)
        Pseudo_tot.generate_target()
        Pseudo_tot.target.shift(3*UP)
        eqn_prd = MathTex(r"P(N) = C \times N")
        
        eqn_term1 = emphasis_box(eqn_prd[0][5])



        self.play(Write(Talking_Text))
        self.wait(3)
        self.play(MoveToTarget(Talking_Text),run_time=1.5)
        self.play(Write(Pseudo_tot))
        self.play(MoveToTarget(Pseudo_tot),FadeOut(Talking_Text),run_time=1.5)
        self.wait(2)
        self.play(Write(eqn_prd),run_time=1.5)
        self.wait(1)
        self.play(Write(eqn_term1[1]))
        self.wait(2)
        self.play(FadeOut(eqn_prd),FadeOut(eqn_term1[1]),FadeOut(Pseudo_tot))
        
        axes = always_redraw(lambda: Axes(
                x_range = [0,12,1],
                x_length = 12,
                y_range = [0,1,0.25],
                y_length = 3,
                axis_config=CONFIG,
                ).to_edge(DL).set_color(GREY)
            )
        labels = axes.get_axis_labels(
            Tex("").scale(0.7), Text("P(N)").scale(0.45)
        )
        
        
        N = ValueTracker(0)
        func_OG = always_redraw(lambda: 
                        axes.plot(lambda x: (C*x),
                                             x_range = [0,N.get_value()], color = YELLOW))
        func = axes.plot(lambda x: C*x, x_range = [0,12], color = YELLOW)

        Dot_x = always_redraw(
            lambda: Dot(color = YELLOW).scale(1).move_to(axes.c2p(N.get_value(), 
                                                    func.underlying_function(N.get_value()))))
        
        lines = axes.get_lines_to_point(axes.c2p(12,1))

        More_Text = Text("Guarenteed bash within 12 hits").next_to(axes,UP)
        
        
        
        
        self.play(Write(axes), Write(labels),run_time=1.5)
        
        self.play(Write(func_OG),Write(Dot_x))
        for ii in range(12):
            self.play(N.animate.increment_value(1),run_time=1)
        self.play(Write(lines))
        self.play(Write(More_Text))
        self.wait(2)

        OutGroup = VGroup(lines,axes,func_OG,Dot_x)
        TextGroup = VGroup(More_Text,Pseudo_tot,labels)
        
        return OutGroup,TextGroup
    
    def prd_Scene3(self,C_pmd, N_pmd, P_pmd, OutGroup,TextGroup):
        N_len = len(N_pmd)
        chart_x_len = 10
        chart_y_len = 3
        box_height = ValueTracker((0.09/0.3) * 3)
        y_midpt = ValueTracker((0.09/0.3) * 0.15)
        x_midpt = ValueTracker(0.5)
        print(N_pmd)
        print(P_pmd)
        chart = BarChart(
            values    = P_pmd,
            bar_names = N_pmd,
            y_range=[0, 0.3,0.1],
            y_length=3,
            x_length=chart_x_len,
            x_axis_config={"font_size": 20},
        ).to_edge(DOWN)
        rect2 = always_redraw(lambda: Rectangle(width=chart_x_len/N_len, 
                                        height=box_height.get_value()).move_to(chart.c2p(x_midpt.get_value(),
                                                                                         y_midpt.get_value())).set_color(YELLOW))
        txt = Text('So does this data appear in practice?').next_to(chart,UP)

        self.play(FadeOut(TextGroup))
        self.play(ReplacementTransform(OutGroup,chart))
        self.wait(2)
        self.play(Write(rect2))
        self.wait(3)
        self.play(box_height.animate.set_value((P_pmd[3]/0.3) * chart_y_len),
                  y_midpt.animate.set_value((P_pmd[3]/0.3) * 0.15),
                  x_midpt.animate.set_value(3.5),run_time=2)
        self.wait(2)
        self.play(Write(txt))
        self.wait(2)
        self.play(FadeOut(txt),FadeOut(chart),FadeOut(rect2))


    def bash_simulation_prd(self,C_pmd):
        N_pmd = [ii+1 for ii in range(12)]
        
        hits_txt = Text('Hits: ').to_edge(UL)
        prob_txt = Text('Prob: ').next_to(hits_txt,DOWN)
        roll_txt = Text('Roll: ').next_to(prob_txt,DOWN)

        
        vector_slots = [0]*len(N_pmd)
        chart = always_redraw( lambda : BarChart(
            values    = vector_slots,
            bar_names = N_pmd,
            y_range=[0, 200, 40],
            y_length=3,
            x_length=12,
            x_axis_config={"font_size": 20},
        ).to_edge(DOWN))
        
        def randomize_number(num):
            value = random.uniform(0,1)
            num.set_value(value)
            
        def check_num(num):
            if num.get_value() <= check:
                num.set_color(GREEN)
            else:
                num.set_color(RED)
                
        def dota_bash_simulator():
            import random
            [C,prob] = pseudorand(0.25)
            
            N = []
            vector_slots = np.zeros(12)
            
            for ii in range(200):
                for n in range(1,100):
                    num = random.uniform(0, 1)
            
                    prob = C*n
                    if num < prob:
                        N.append(n)
                        break
            
                vector_slots[N[ii]-1] += 1
        
            return vector_slots, N
        
        
        text_100 = Text('Try 200x').next_to(chart,UP)
        
        self.play(Write(hits_txt),Write(roll_txt),Write(prob_txt),Write(chart))
        
        
        
        for jj in range(1):
            hit_num = 1
            hit = DecimalNumber(hit_num).next_to(hits_txt,RIGHT)
            hitters =VGroup()
            hitters.add(hit)
            
            num = DecimalNumber().next_to(roll_txt,RIGHT)
            numbers =VGroup()
            numbers.add(num)
            check = C_pmd*(1)
            
            prob_num = DecimalNumber(check).next_to(prob_txt,RIGHT)
            probability =VGroup()
            probability.add(prob_num)
            
            chart_old = chart
            for ii in range(12):
                self.play(Write(hitters[ii]),Write(numbers[ii]),Write(probability[ii]))
                self.play(UpdateFromFunc(numbers[ii],randomize_number),run_time=0.3)
                self.play(UpdateFromFunc(numbers[ii],check_num),run_time=0.3)
                self.wait(0.2)
                if numbers[ii].get_value() <= check:
                    break
                else:
                    check = check + C_pmd
                    hit_num = hit_num + 1
                    hit = DecimalNumber(hit_num).next_to(hitters[ii],RIGHT)
                    hitters.add(hit)
                    
                    num = DecimalNumber().next_to(numbers[ii],RIGHT)
                    numbers.add(num)
                    
                    prob_num = DecimalNumber(check).next_to(probability[ii],RIGHT)
                    probability.add(prob_num)
            self.play(Write(text_100))
            
            vector_slots_dum,asdasd = dota_bash_simulator()
            vector_slots = list(np.array(vector_slots) + np.array(vector_slots_dum))
            
            chart = always_redraw( lambda : BarChart(
                values    = vector_slots,
                bar_names = N_pmd,
                y_range=[0, 200, 40],
                y_length=3,
                x_length=12,
                x_axis_config={"font_size": 20},
            ).to_edge(DOWN))
            
            self.play(ReplacementTransform(chart_old,chart))
            self.wait(1)
            self.play(FadeOut(text_100), FadeOut(numbers), FadeOut(hitters), FadeOut(probability),run_time = 0.5)




    def construct(self):
# =============================================================================
#         img1, img2, txt = self.basher_setup_scene(True)
#         self.play(FadeOut(img1,img2,txt))
#         plotter_func = self.first_number_scene(True)
#         Scene_tot_unifpmd = self.uniform_chart(plotter_func,True)
#         
#         self.play(FadeOut(Scene_tot_unifpmd))
#         self.wait(2)
# =============================================================================
        [C_pmd,N_pmd,P_pmd] = self.prd_pmd(0.25)
        #OutGroup,TextGroup = self.prd_Scene1(C_pmd)
        #self.prd_Scene3(C_pmd, N_pmd, P_pmd, OutGroup,TextGroup)
        self.bash_simulation_prd(C_pmd)
