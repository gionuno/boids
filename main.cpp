#include <GL/glew.h>
#include <GL/gl.h>

#include "shader.hpp"
#include "texture.hpp"
#include "mesh.hpp"
#include "mpnn.hpp"
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>

#include <cmath>

#define ARMA_DEBUG

using namespace std;
using namespace arma;

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

int win_x = 640;
int win_y = 640;

GLFWwindow * window = nullptr;

struct boids
{
	
	vector<vec> X;
	vector<vec> dX;
	vector<int> type;
	
    kd_tree * tree;
    int N,K,K2;
    boids(int N_,int K_,int K2_)
    {
        N = N_;
        K = K_;
        K2 = K2_;
        for(int n=0;n<N;n++)
        {
			X.push_back(150.0*randu<vec>(2)-75.0);
			dX.push_back(2.0*randu<vec>(2)-1.0);
			type.push_back((randu<double>() <= 0.95) ? 0 : 1);
	    }
        tree = new kd_tree(ones(2),zeros<ivec>(2),-100.0*ones(2),100.*ones(2));
    }
    vec within_bounds(vec2 x)
    {
        if(norm(x)>100.0)
            return -x;
        else return zeros<vec>(x.n_elem);
    }
    vec within_speed(vec2 v)
    {
		if(norm(v) < 0.1)
			return 0.5*v;
		else if(norm(v) > 100.0)
			return -0.5*v;
		else
			return zeros<vec>(v.n_elem);
		
		
	}
    void step(double dt)
    {
        tree->set(X);
        uvec idx = zeros<uvec>(K+K2+1);
        int n_N = 0;
        int s_N = 0;
        vec a = zeros<vec>(2);
        vec s = zeros<vec>(2);
        vec c = zeros<vec>(2);
        
        mat a_m = zeros<mat>(2,2);
        mat c_m = zeros<mat>(2,2);
		vec n_M = zeros<vec>(2);
		for(int n=0;n<N;n++)
        {
			n_M(type[n]) += 1.0;
			
            c_m.col(type[n]) += X[n];
            a_m.col(type[n]) += dX[n];
        }

        for(int n=0;n<N;n++)
        {
            tree->searchK(X[n],idx);
            a = zeros<vec>(2);
            s = zeros<vec>(2);
            c = zeros<vec>(2);
            n_N = 0;
            s_N = 0;
            for(int k=1;k<=K;k++)
            {
				bool same_t = (type[n] == 1 ? true : type[idx(k)] == type[n]);
				double ang  = (atan2(X[idx(k)](1)-X[n](1),X[idx(k)](0)-X[n](0))-atan2(dX[n](1),dX[n](0)))/datum::pi;
                double dist = tree->dist_to_point(X[idx(k)],X[n]);
                if(dist < 5.0)
                {
                    s += X[n]-X[idx(k)];
                    s_N++;
                }
                if(-0.8 < ang && ang < 0.8)
                {
                    a += (type[n] == 0 ? (type[idx(k)] == 0? 1.0 : -20.0) : (type[idx(k)] == 0? 10.0 : -1.0))*(dX[idx(k)]-dX[n]);
                    c += (type[n] == 0 ? (type[idx(k)] == 0? 1.0 : -20.0) : (type[idx(k)] == 0? 10.0 : -1.0))*(X[idx(k)]-X[n]);
                    n_N++;
				}
            }
            for(int k=K+1;k<=K+K2;k++)
            {
				bool same_t = (type[n] == 1 ? true : type[idx(k)] == type[n]);
                double ang  = (atan2(X[idx(k)](1)-X[n](1),X[idx(k)](0)-X[n](0))-atan2(dX[n](1),dX[n](0)))/datum::pi;
                double dist = tree->dist_to_point(X[idx(k)],X[n]);
                if(dist < 5.0)
                {
                    s += X[n]-X[idx(k)];
                    s_N++;
                }
				else
				{
					if(-0.9 < ang && ang < 0.9)
					{
						a += (type[n] == 0 ? (type[idx(k)] == 0? 1.0 : -15.0) : (type[idx(k)] == 0? 5.0 : -1.0))*(dX[idx(k)]-dX[n]);
						c += (type[n] == 0 ? (type[idx(k)] == 0? 1.0 : -15.0) : (type[idx(k)] == 0? 5.0 : -1.0))*(X[idx(k)]-X[n]);
						n_N++;
					}
				}
            }
            vec rule1 = within_bounds(X[n]);
            vec rule2 = within_speed(dX[n]);
            dX[n] = dX[n] 
					+ 10.*rule1 
					+ 1.5*rule2
					+ 0.05*((a_m.col(type[n])-dX[n])/(n_M(type[n])-1+0.001)-dX[n]) 
					+ 0.01*((c_m.col(type[n])-X[n])/(n_M(type[n])-1+0.001)-X[n])
					+ 0.1*(a/(n_N+0.001))
					+ 0.1*(c/(n_N+0.001))
            		+ 1.0*(s/(s_N+0.001));
			//if(norm(dX[n])>100.0)
			//	dX[n] *= 0.1;
		}

        for(int n=0;n<N;n++)
        {
            X[n] = X[n]+dt*dX[n];
            dX[n] *= 0.95;
        }
    }
};

int init_gl_window()
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(win_x,win_y,"Boids",nullptr,nullptr);

    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glewInit();

    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

    return 0;
}


int main(int argc,char ** argv)
{
    srand(time(nullptr));
    if(init_gl_window()<0) return -1;

    boid_fig  boid_;

    int N  = 250;
    int K  = 5;
    int K2 = 5;
    boids B(N,K,K2);

    shader    show_;

    show_.load_file(GL_VERTEX_SHADER  ,"vert.glsl");
    show_.load_file(GL_FRAGMENT_SHADER,"frag.glsl");
    show_.create();
    vec c1 = zeros<vec>(3);
    c1(0) = 0.3;
    c1(1) = 0.3;
    c1(2) = 1.0;
    vec c2 = zeros<vec>(3);
    c2(0) = 1.0;
    c2(1) = 0.0;
    c2(2) = 0.0;

    double t = 0.;
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwGetFramebufferSize(window, &win_x, &win_y);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        show_.begin();
        for(int n=0;n<N;n++)
        {
            glUniform2f(show_("pos"),B.X[n](0)/100.0,B.X[n](1)/100.0);
            glUniform1f(show_("ang"),atan2(B.dX[n](1),B.dX[n](0)));
            if(B.type[n] == 0)
				glUniform3f(show_("col"),c1(0),c1(1),c1(2));
            else
            	glUniform3f(show_("col"),c2(0),c2(1),c2(2));
            boid_.draw();
        }
        show_.end();
        B.step(1e-2);
        t += 1.0;
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
