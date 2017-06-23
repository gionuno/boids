#ifndef MPNN_HPP_INCLUDED
#define MPNN_HPP_INCLUDED

#include <armadillo>
#include <vector>

using namespace std;
using namespace arma;

const uint max_N = 10;

struct kd_node
{
    kd_node * l;
    kd_node * r;

    uint   s_d;

    vec     l_x;
    vec     h_x;

    vector<uword> idxs;

    kd_node()
    {
        l = nullptr;
        r = nullptr;
    }
    kd_node(const vector<uword> & idxs_)
    {
        l = nullptr;
        r = nullptr;

        idxs = idxs_;
    }

    ~kd_node()
    {
        idxs.clear();
        if(l) delete l;
        if(r) delete r;
    }

};

struct kd_tree
{
    vec  mu;
    ivec topo;

    vector<vec> x;

    kd_node * root;

    vec L;
    vec H;

    kd_tree(const vec & mu_,
            const ivec & topo_,
            const vec  & l_bounds,
            const vec  & h_bounds)
    {
        mu   = mu_;
        topo = topo_;

        root = new kd_node();

        L = l_bounds;
        H = h_bounds;

        root->l_x = L;
        root->h_x = H;
    }

    ~kd_tree()
    {
        x.clear();
        if(root) delete root;
    }

    void set(const vector<vec> & x_)
    {
        vector<uword> idxs_;
        for(uword n=0;n<x_.size();n++)
        {
            if(n < x.size())
                x[n] = x_[n];
            else
                x.push_back(x_[n]);
            idxs_.push_back(n);
        }
        if(root) delete root;
        root = new kd_node(idxs_);
        root->l_x = L;
        root->h_x = H;
        balance(root);
    }

    double r_norm(double a,double b)
    {
        return fabs(a-b);
    }
    double s_norm(double a,double b)
    {
        return std::min(1.0-fabs(a-b),fabs(a-b));
    }

    double dist_to_point(const vec & a,const vec & b)
    {
        double D = 0.0;
        for(uint d=0;d<topo.n_elem;d++)
        {
            if(topo(d)==0)
                D += mu(d)*pow(r_norm(a(d),b(d)),2.0);
            else
                D += mu(d)*pow(s_norm(a(d),b(d)),2.0);
        }
        return D;
    }

    double dist_to_rect(const vec & a,const vec & l,const vec & h)
    {
        double D = 0.0;
        for(uint d=0;d<topo.n_elem;d++)
        {
            if(topo(d) == 0)
            {
                if(a(d)<l(d))
                    D += mu(d)*pow(l(d)-a(d),2.0);
                else if(a(d)>h(d))
                    D += mu(d)*pow(a(d)-h(d),2.0);
            }
            else if(topo(d) == 1)
            {
                if(a(d)<l(d))
                    D += mu(d)*pow(std::min(l(d)-a(d),1.0-fabs(h(d)-a(d))),2.0);
                else if(a(d)>h(d))
                    D += mu(d)*pow(std::min(a(d)-h(d),1.0-fabs(l(d)-a(d))),2.0);
            }
        }
        return D;
    }

    void balance(kd_node * & node)
    {
        if(node == nullptr) return;

        if(node->idxs.size() >= max_N)
        {

            vec m = x[node->idxs[0]];
            vec M = x[node->idxs[0]];
            for(auto & i : node->idxs)
            {
                m = arma::min(m,x[i]);
                M = arma::max(M,x[i]);
            }

            vec r = M-m;
            uword s_d=0;
            for(uword d=0;d<topo.n_elem;d++)
                if(r(d) > r(s_d)) s_d = d;

            double s_x = 0.5*r(s_d)+m(s_d);

            vector<uword> l_idxs;
            vector<uword> r_idxs;

            for(auto & i : node->idxs)
            {
                if(x[i](s_d) < s_x)
                    l_idxs.push_back(i);
                else
                    r_idxs.push_back(i);
            }

            if(l_idxs.size() > 0)
            {
                node->l = new kd_node(l_idxs);
                node->l->l_x = node->l_x;
                node->l->h_x = node->h_x;
                node->l->h_x(s_d) = s_x;
            }
            if(r_idxs.size() > 0)
            {
                node->r = new kd_node(r_idxs);
                node->r->l_x = node->l_x;
                node->r->h_x = node->h_x;
                node->r->l_x(s_d) = s_x;
            }
            node->idxs.clear();
        }
        if(node->l)
            balance(node->l);
        if(node->r)
            balance(node->r);
    }

    void insert(vec y)
    {
        x.push_back(y);
        insert(root,y,x.size()-1);
        balance(root);
    }

    void insert(kd_node * & node,const vec & y,uword idx)
    {
        if(node == nullptr)
        {
            node = new kd_node();
            node->idxs.push_back(idx);
            return;
        }
        if(node->idxs.size()<max_N && !node->l && !node->r)
        {
            node->idxs.push_back(idx);
            return;
        }

        double d_l = datum::inf;
        double d_r = datum::inf;
        if(node->l)
            d_l = dist_to_rect(y,node->l->l_x,node->l->h_x);
        if(node->r)
            d_r = dist_to_rect(y,node->r->l_x,node->r->h_x);
        if(d_l < d_r)
            insert(node->l,y,idx);
        else
            insert(node->r,y,idx);
    }

    void search(const vec & y,uword & b_idx)
    {
        b_idx = 0;
        double b_dist = datum::inf;
        search(root,y,b_idx,b_dist);
    }

    void search(kd_node * & node,const vec & y,uword & b_idx,double & b_dist)
    {
        if(node == nullptr) return;

        double box_dist = dist_to_rect(y,node->l_x,node->h_x);
        if(box_dist < b_dist)
        {
            for(auto i : node->idxs)
            {
                double aux = dist_to_point(x[i],y);
                if(aux < b_dist)
                {
                    b_dist = aux;
                    b_idx  = i;
                }
            }

            double d_l = datum::inf;
            double d_r = datum::inf;
            if(node->l)
                d_l = dist_to_rect(y,node->l->l_x,node->l->h_x);
            if(node->r)
                d_r = dist_to_rect(y,node->r->l_x,node->r->h_x);
            if(d_l < d_r)
            {
                if(node->l) search(node->l,y,b_idx,b_dist);
                if(node->r) search(node->r,y,b_idx,b_dist);
            }
            else
            {
                if(node->r) search(node->r,y,b_idx,b_dist);
                if(node->l) search(node->l,y,b_idx,b_dist);
            }
        }
    }

    void searchK(const vec & y,uvec & b_idx)
    {
        for(uword k=0;k<b_idx.n_elem;k++)
            b_idx(k) = 0;
        vec b_dist = datum::inf*ones(b_idx.n_elem);
        searchK(root,y,b_idx,b_dist);
    }

    void searchK(kd_node * & node,const vec & y,uvec & b_idx,vec & b_dist)
    {
        if(node == nullptr) return;

        double box_dist = dist_to_rect(y,node->l_x,node->h_x);
        double b_max = b_dist(b_idx.n_elem-1);
        if(box_dist < b_max)
        {
            for(auto i : node->idxs)
            {
                double aux = dist_to_point(x[i],y);
                uvec a_idx = zeros<uvec>(b_idx.n_elem+1);
                vec a_dist = zeros<vec>(b_idx.n_elem+1);
                for(uword k=0;k<b_idx.n_elem;k++)
                {
                    a_idx(k) = b_idx(k);
                    a_dist(k) = b_dist(k);
                }
                a_idx(a_idx.n_elem-1) = i;
                a_dist(a_idx.n_elem-1) = aux;
                uvec a = sort_index(a_dist);
                for(uword k=0;k<b_idx.n_elem;k++)
                {
                    b_idx(k) = a_idx(a(k));
                    b_dist(k) = a_dist(a(k));
                }
            }
            double d_l = datum::inf;
            double d_r = datum::inf;
            if(node->l)
                d_l = dist_to_rect(y,node->l->l_x,node->l->h_x);
            if(node->r)
                d_r = dist_to_rect(y,node->r->l_x,node->r->h_x);
            if(d_l < d_r)
            {
                if(node->l) searchK(node->l,y,b_idx,b_dist);
                if(node->r) searchK(node->r,y,b_idx,b_dist);
            }
            else
            {
                if(node->r) searchK(node->r,y,b_idx,b_dist);
                if(node->l) searchK(node->l,y,b_idx,b_dist);
            }
        }
    }


};


#endif // MPNN_HPP_INCLUDED
