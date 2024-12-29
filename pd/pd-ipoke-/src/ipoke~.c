//	ipoke~ - a buffer writter with skipped address filling feature (interpolated or not)
//	by Pierre Alexandre Tremblay
//	v.2 optimised at the University of Huddersfield, with the help of the dev@cycling74.com comments of Mark Pauley, 
//          Alex Harker, Ben Neville and Peter Castine, addressed  to me or to others.
//	v.3 updated for Max5
//  Pd port by Katja Vetter 2012, with assistance from Matt Barber, Julian Brooks, Alexander Harker, Charles Henry 
//          and P.A. Tremblay.

#include <stdbool.h>
#include "m_pd.h"

#define MAJOR_VERSION  0
#define MINOR_VERSION  5
#define BUGFIX_VERSION 0

#define CLIP(a, lo, hi) ( (a)>(lo)?( (a)<(hi)?(a):(hi) ):(lo) )     // as in MaxMsp's ext_common.h

void *ipoke_class;

typedef struct _ipoke
{
    t_object l_obj;
    t_word *l_buf;              // pointer to external array
    t_symbol *l_sym;            // pointer to struct holding the name of external array
    long bufsize;
	bool l_interp;
	bool l_overdub;
	long l_index_precedent;
	long l_nb_val;
	t_float l_valeur;
    t_float l_f;                // for MAINSIGNALIN
} t_ipoke;

t_symbol *ps_buffer;

static void *ipoke_new(t_symbol *s)
{
	t_ipoke *x = (t_ipoke*)pd_new(ipoke_class);
    inlet_new(&x->l_obj, &x->l_obj.ob_pd, &s_signal, &s_signal);   // inlet for float indices
	x->l_sym = s;
	x->l_interp = 1;
	x->l_overdub = 0;
	x->l_index_precedent = -1;

	return (x);
}

static inline long wrap_index(size_t index, size_t arrayLength)
{
	while(index >= arrayLength)
        index -= arrayLength;
    return index;
}

// four different subroutines of perform method:
// ipoke_perform_basic() - interpolation no, overdub no
// ipoke_perform_interp() - interpolation yes, overdub no
// ipoke_perform_overdub() - interpolation no, overdub yes
// ipoke_perform_interp_overdub() - interpolation yes, overdub yes

static void ipoke_perform_basic(t_int *w, t_ipoke *x)    // perform method without interpolation and overdub
{
    t_float *inval = (t_float *)(w[2]);
    t_float *inind = (t_float *)(w[3]);
	int n = (int)(w[4]);
    
	t_float valeur_entree, valeur, index_tampon, coeff;
	long bufsize, nb_val, index=0, index_precedent, pas, i, demivie;
    
    t_word *tab = x->l_buf;
    bufsize = x->bufsize;
	demivie = (long)(bufsize * 0.5);
	index_precedent = x->l_index_precedent;
	valeur = x->l_valeur;
	nb_val = x->l_nb_val;
    
    while (n--)	// dsp loop without interpolation
    {
        valeur_entree = *inval++;
        index_tampon = *inind++;
        
        if (index_tampon < 0.0)											// if the writing is stopped
        {
            if (index_precedent >= 0)									// and if it is the 1st one to be stopped
            {
                tab[index_precedent].w_float = valeur/nb_val;           // write the average value at the last given index
                valeur = 0.0;
                index_precedent = -1;
            }
        }			
        else                                                            // if writing
        {
            index = wrap_index((long)(index_tampon + 0.5),bufsize);		// round the next index and make sure he is in the buffer's boundaries
            
            if (index_precedent < 0)									// if it is the first index to write, resets the averaging and the values
            {
                index_precedent = index;
                nb_val = 0;
            }
            
            if (index == index_precedent)								// if the index has not moved, accumulate the value to average later.
            {
                valeur += valeur_entree;
                nb_val += 1;
            }
            else														// if it moves
            {
                if (nb_val != 1)										// is there more than one values to average
                {
                    valeur = valeur/nb_val;								// if yes, calculate the average
                    nb_val = 1;
                }
                
                tab[index_precedent].w_float = valeur;                  // write the average value at the last index
                
                pas = index - index_precedent;							// calculate the step to do
                
                if (pas > 0)											// are we going up
                {
                    if (pas > demivie)									// is it faster to go the other way round?
                    {
                        for(i=(index_precedent-1);i>=0;i--)				// fill the gap to zero
                            tab[i].w_float = valeur;
                        for(i=(bufsize-1);i>index;i--)					// fill the gap from the top
                            tab[i].w_float = valeur;
                    }
                    else												// if not, just fill the gaps
                    {
                        for (i=(index_precedent+1); i<index; i++)
                            tab[i].w_float = valeur;
                    }
                }
                else													// if we are going down
                {
                    if ((-pas) > demivie)								// is it faster to go the other way round?
                    {
                        for(i=(index_precedent+1);i<bufsize;i++)		// fill the gap to the top
                            tab[i].w_float = valeur;
                        for(i=0;i<index;i++)							// fill the gap from zero
                            tab[i].w_float = valeur;
                    }
                    else												// if not, just fill the gaps
                    {
                        for (i=(index_precedent-1); i>index; i--)
                            tab[i].w_float = valeur;
                    }
                }
                
                valeur = valeur_entree;									// transfer the new previous value
            }// end of else (if it moves)
        }// end of else (if writing)	
        index_precedent = index;										// transfer the new previous address
    }// end of while(n--)
    
    x->l_index_precedent = index_precedent;
	x->l_valeur = valeur;
	x->l_nb_val = nb_val; 
}

static void ipoke_perform_interp(t_int *w, t_ipoke *x) // perform method with interpolation, without overdub
{
    t_float *inval = (t_float *)(w[2]);
    t_float *inind = (t_float *)(w[3]);
	int n = (int)(w[4]);
    
	t_float valeur_entree, valeur, index_tampon, coeff;
	long bufsize, nb_val, index=0, index_precedent, pas, i, demivie;
    
    t_word *tab = x->l_buf;
    bufsize = x->bufsize;
	demivie = (long)(bufsize * 0.5);
	index_precedent = x->l_index_precedent;
	valeur = x->l_valeur;
	nb_val = x->l_nb_val;
    
    while (n--)	// dsp loop with interpolation
    {
        valeur_entree = *inval++;
        index_tampon = *inind++;
        
        if (index_tampon < 0.0)											// if the writing is stopped
        {
            if (index_precedent >= 0)									// and if it is the 1st one to be stopped
            {
                tab[index_precedent].w_float = valeur/nb_val;           // write the average value at the last given index
                valeur = 0.0;
                index_precedent = -1;
            }
        }			
        else                                                            // if writing
        {
            index = wrap_index((long)(index_tampon + 0.5),bufsize);		// round the next index and make sure he is in the buffer's boundaries
            
            if (index_precedent < 0)									// if it is the first index to write, resets the averaging and the values
            {
                index_precedent = index;
                nb_val = 0;
            }
            
            if (index == index_precedent)								// if the index has not moved, accumulate the value to average later.
            {
                valeur += valeur_entree;
                nb_val += 1;
            }
            else														// if it moves
            {
                if (nb_val != 1)										// is there more than one values to average
                {
                    valeur = valeur/nb_val;								// if yes, calculate the average
                    nb_val = 1;
                }
                
                tab[index_precedent].w_float = valeur;                  // write the average value at the last index
                
                pas = index - index_precedent;							// calculate the step to do
                
                if (pas > 0)											// are we going up
                {
                    if (pas > demivie)									// is it faster to go the other way round?
                    {
                        pas -= bufsize;									// calculate the new number of steps
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        
                        for(i=(index_precedent-1);i>=0;i--)				// fill the gap to zero
                        {
                            valeur -= coeff;
                            tab[i].w_float = valeur;
                        }
                        for(i=(bufsize-1);i>index;i--)					// fill the gap from the top
                        {
                            valeur -= coeff;
                            tab[i].w_float = valeur;
                        }
                    }
                    else												// if not, just fill the gaps
                    {
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        for (i=(index_precedent+1); i<index; i++)
                        {
                            valeur += coeff;
                            tab[i].w_float = valeur;
                        }
                    }
                }
                else													// if we are going down
                {
                    if ((-pas) > demivie)								// is it faster to go the other way round?
                    {
                        pas += bufsize;									// calculate the new number of steps
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        
                        for(i=(index_precedent+1);i<bufsize;i++)		// fill the gap to the top
                        {
                            valeur += coeff;
                            tab[i].w_float = valeur;
                        }
                        for(i=0;i<index;i++)							// fill the gap from zero
                        {
                            valeur += coeff;
                            tab[i].w_float = valeur;
                        }
                    }
                    else												// if not, just fill the gaps
                    {
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        for (i=(index_precedent-1); i>index; i--)
                        {
                            valeur -= coeff;
                            tab[i].w_float = valeur;
                        }							
                    }
                }
                
                valeur = valeur_entree;									// transfer the new previous value
            }// end of else (if it moves)
        }// end of else (if writing)	
        index_precedent = index;										// transfer the new previous address
    }// end of while(n--)
    
    x->l_index_precedent = index_precedent;
	x->l_valeur = valeur;
	x->l_nb_val = nb_val;
}

static void ipoke_perform_overdub(t_int *w, t_ipoke *x)    // perform method with overdub, no interpolation
{
    t_float *inval = (t_float *)(w[2]);
    t_float *inind = (t_float *)(w[3]);
	int n = (int)(w[4]);
    
	t_float valeur_entree, valeur, index_tampon, coeff;
	long bufsize, nb_val, index=0, index_precedent, pas, i, demivie;
    
    t_word *tab = x->l_buf;
    bufsize = x->bufsize;
	demivie = (long)(bufsize * 0.5);
	index_precedent = x->l_index_precedent;
	valeur = x->l_valeur;
	nb_val = x->l_nb_val;
    
    while (n--)	// dsp loop without interpolation
    {
        valeur_entree = *inval++;
        index_tampon = *inind++;
        
        if (index_tampon < 0.0)											// if the writing is stopped
        {
            if (index_precedent >= 0)									// and if it is the 1st one to be stopped
            {
                tab[index_precedent].w_float += valeur/nb_val;          // write the average value at the last given index
                valeur = 0.0;
                index_precedent = -1;
            }
        }			
        else                                                            // if writing
        {
            index = wrap_index((long)(index_tampon + 0.5),bufsize);		// round the next index and make sure he is in the buffer's boundaries
            
            if (index_precedent < 0)									// if it is the first index to write, resets the averaging and the values
            {
                index_precedent = index;
                nb_val = 0;
            }
            
            if (index == index_precedent)								// if the index has not moved, accumulate the value to average later.
            {
                valeur += valeur_entree;
                nb_val += 1;
            }
            else														// if it moves
            {
                if (nb_val != 1)										// is there more than one values to average
                {
                    valeur = valeur/nb_val;								// if yes, calculate the average
                    nb_val = 1;
                }
                
                tab[index_precedent].w_float += valeur;                 // write the average value at the last index
                
                pas = index - index_precedent;							// calculate the step to do
                
                if (pas > 0)											// are we going up
                {
                    if (pas > demivie)									// is it faster to go the other way round?
                    {
                        for(i=(index_precedent-1);i>=0;i--)				// fill the gap to zero
                            tab[i].w_float += valeur;
                        for(i=(bufsize-1);i>index;i--)					// fill the gap from the top
                            tab[i].w_float += valeur;
                    }
                    else												// if not, just fill the gaps
                    {
                        for (i=(index_precedent+1); i<index; i++)
                            tab[i].w_float += valeur;
                    }
                }
                else													// if we are going down
                {
                    if ((-pas) > demivie)								// is it faster to go the other way round?
                    {
                        for(i=(index_precedent+1);i<bufsize;i++)		// fill the gap to the top
                            tab[i].w_float += valeur;
                        for(i=0;i<index;i++)							// fill the gap from zero
                            tab[i].w_float += valeur;
                    }
                    else												// if not, just fill the gaps
                    {
                        for (i=(index_precedent-1); i>index; i--)
                            tab[i].w_float += valeur;
                    }
                }
                
                valeur = valeur_entree;									// transfer the new previous value
            }// end of else (if it moves)
        }// end of else (if writing)	
        index_precedent = index;										// transfer the new previous address
    }// end of while(n--)
    
    x->l_index_precedent = index_precedent;
	x->l_valeur = valeur;
	x->l_nb_val = nb_val;
}

static void ipoke_perform_interp_overdub(t_int *w, t_ipoke *x) // perform method with interpolation and overdub
{
    t_float *inval = (t_float *)(w[2]);
    t_float *inind = (t_float *)(w[3]);
	int n = (int)(w[4]);
    
	t_float valeur_entree, valeur, index_tampon, coeff;
	long bufsize, nb_val, index=0, index_precedent, pas, i, demivie;
    
    t_word *tab = x->l_buf;
    bufsize = x->bufsize;
	demivie = (long)(bufsize * 0.5);
	index_precedent = x->l_index_precedent;
	valeur = x->l_valeur;
	nb_val = x->l_nb_val;
    
    while (n--)	// dsp loop with interpolation
    {
        valeur_entree = *inval++;
        index_tampon = *inind++;
        
        if (index_tampon < 0.0)											// if the writing is stopped
        {
            if (index_precedent >= 0)									// and if it is the 1st one to be stopped
            {
                tab[index_precedent].w_float += valeur/nb_val;          // write the average value at the last given index
                valeur = 0.0;
                index_precedent = -1;
            }
        }
        else                                                            // if writing
        {
            index = wrap_index((long)(index_tampon + 0.5),bufsize);		// round the next index and make sure he is in the buffer's boundaries
            
            if (index_precedent < 0)									// if it is the first index to write, resets the averaging and the values
            {
                index_precedent = index;
                nb_val = 0;
            }
            
            if (index == index_precedent)								// if the index has not moved, accumulate the value to average later.
            {
                valeur += valeur_entree;
                nb_val += 1;
            }
            else														// if it moves
            {
                if (nb_val != 1)										// is there more than one values to average
                {
                    valeur = valeur/nb_val;								// if yes, calculate the average
                    nb_val = 1;
                }
                
                tab[index_precedent].w_float += valeur;                 // write the average value at the last index
                
                pas = index - index_precedent;							// calculate the step to do
                
                if (pas > 0)											// are we going up
                {
                    if (pas > demivie)									// is it faster to go the other way round?
                    {
                        pas -= bufsize;									// calculate the new number of steps
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        
                        for(i=(index_precedent-1);i>=0;i--)				// fill the gap to zero
                        {
                            valeur -= coeff;
                            tab[i].w_float += valeur;
                        }
                        for(i=(bufsize-1);i>index;i--)					// fill the gap from the top
                        {
                            valeur -= coeff;
                            tab[i].w_float += valeur;
                        }
                    }
                    else												// if not, just fill the gaps
                    {
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        for (i=(index_precedent+1); i<index; i++)
                        {
                            valeur += coeff;
                            tab[i].w_float += valeur;
                        }
                    }
                } // end if we are going up
                else													// if we are going down
                {
                    if ((-pas) > demivie)								// is it faster to go the other way round?
                    {
                        pas += bufsize;									// calculate the new number of steps
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        
                        for(i=(index_precedent+1);i<bufsize;i++)		// fill the gap to the top
                        {
                            valeur += coeff;
                            tab[i].w_float += valeur;
                        }
                        for(i=0;i<index;i++)							// fill the gap from zero
                        {
                            valeur += coeff;
                            tab[i].w_float += valeur;
                        }
                    }
                    else												// if not, just fill the gaps
                    {
                        coeff = (valeur_entree - valeur) / pas;			// calculate the interpolation coefficient
                        for (i=(index_precedent-1); i>index; i--)
                        {
                            valeur -= coeff;
                            tab[i].w_float += valeur;
                        }							
                    }
                }// end if we are going down
                
                valeur = valeur_entree;									// transfer the new previous value
            }// end of else (if it moves)
        }// end of else (if writing)
        index_precedent = index;										// transfer the new previous address
    }// end of while(n--)
    
    x->l_index_precedent = index_precedent;
	x->l_valeur = valeur;
	x->l_nb_val = nb_val;
}

static t_int *ipoke_perform(t_int *w)
{
    t_ipoke *x = (t_ipoke *)(w[1]);
    
    if(!x->l_buf) goto out;                           // skip dsp loop if buffer does not exist
    
	int interp = x->l_interp;
	int overdub = x->l_overdub;
	
    if(interp)
    {
        if(!overdub) ipoke_perform_interp(w, x);      // interpolation yes, overdub no (default)
        else ipoke_perform_interp_overdub(w, x);      // interpolation yes, overdub yes
    }
    else if(overdub) ipoke_perform_overdub(w, x);     // interpolation no, overdub yes
    else ipoke_perform_basic(w, x);                   // interpolation no, overdub no

out:
	return (w + 5);
}

void ipoke_set(t_ipoke *x, t_symbol *s)
{
	t_garray *b;
    int bufsize;
    x->l_sym = s;
    
    if(!(b = (t_garray*)pd_findbyclass(x->l_sym, garray_class)))
    {
        if(*x->l_sym->s_name) pd_error(x, "ipoke~: %s, no such array", x->l_sym->s_name);
        x->l_buf = 0;
    }
    
    else if (!garray_getfloatwords(b, &bufsize, &x->l_buf))
    {
        pd_error(x, "%s: bad template for ipoke~", x->l_sym->s_name);
        x->l_buf = 0;
    }
    
    else 
    {
        x->bufsize = bufsize;
        garray_usedindsp(b);
    }
}

void ipoke_dsp(t_ipoke *x, t_signal **sp)
{
	x->l_index_precedent = -1;
    ipoke_set(x, x->l_sym);
    dsp_add(ipoke_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
    
}

static void ipoke_interp_setting(t_ipoke *x, t_floatarg interp)
{
	int n = (int)interp;
    
    switch (n)
	{
		case 0:
			x->l_interp = 0;
			break;
		case 1:
			x->l_interp = 1;
			break;
		default:
			pd_error(x, "wrong interpolation type");
			break;
	}
}

static void ipoke_overdub_setting(t_ipoke *x, t_floatarg overdub)
{
	int n = (int)overdub;
    
    switch (n)
	{
		case 0:
			x->l_overdub = 0;
			break;
		case 1:
			x->l_overdub = 1;
			break;
		default:
			pd_error(x, "wrong overdubbing type");
			break;
	}
}

static void ipoke_bang(t_ipoke *x)
{
    t_garray *buf = (t_garray *)pd_findbyclass(x->l_sym, garray_class);
    if(!buf) pd_error(x, "ipoke~: %s, no such array", x->l_sym->s_name);
    else garray_redraw(buf); 
}

void ipoke_tilde_setup(void)
{
    ipoke_class = class_new(gensym("ipoke~"), (t_newmethod)ipoke_new, 0,
        sizeof(t_ipoke), 0, A_DEFSYM, 0);
    CLASS_MAINSIGNALIN(ipoke_class, t_ipoke, l_f);
    class_addmethod(ipoke_class, (t_method)ipoke_dsp, gensym("dsp"), 0);
    class_addmethod(ipoke_class, (t_method)ipoke_interp_setting, gensym("interp"), A_FLOAT, 0);
    class_addmethod(ipoke_class, (t_method)ipoke_overdub_setting, gensym("overdub"), A_FLOAT, 0);
    class_addmethod(ipoke_class, (t_method)ipoke_set, gensym("set"), A_SYMBOL,0);
    class_addbang(ipoke_class, ipoke_bang);
    post("[ipoke~] %d.%d.%d: Pd port of MaxMsp class by P.A. Tremblay", MAJOR_VERSION, MINOR_VERSION, BUGFIX_VERSION);
}
