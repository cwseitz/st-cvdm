import pandas as pd
import matplotlib.pyplot as plt
from cvdm.make import BasicKDE
from cvdm.utils.errors import get_errors
import numpy as np

path='/N/slate/cwseitz/cvdm/Nup96/4x/sum5/'
spots_mle=pd.read_csv(path+'Nup96-LR-crop/Nup96-LR-crop_spots.csv')
spots_mle=spots_mle.loc[spots_mle['frame']%2==0]
spots_cvdm=pd.read_csv(path+'Nup96-SR/Nup96-SR_spots.csv')
xmin,xmax=8,58
ymin,ymax=8,58
spots_cvdm['y_mle']=-0.25+spots_cvdm['y_mle']/4
spots_cvdm['x_mle']=0.5+spots_cvdm['x_mle']/4
spots_mle=spots_mle[(spots_mle['x_mle']>xmin)&(spots_mle['x_mle']<xmax)]
spots_mle=spots_mle[(spots_mle['y_mle']>ymin)&(spots_mle['y_mle']<ymax)]
spots_cvdm=spots_cvdm[(spots_cvdm['x_mle']>xmin)&(spots_cvdm['x_mle']<xmax)]
spots_cvdm=spots_cvdm[(spots_cvdm['y_mle']>ymin)&(spots_cvdm['y_mle']<ymax)]
spots_mle = spots_mle.loc[spots_mle['conv'] == True]

fig,ax=plt.subplots(figsize=(3.5,3.5))
pixel_size=0.025 #um
theta_mle=spots_mle[['x_mle','y_mle','peak','N0']].values
theta_cvdm=spots_cvdm[['x_mle','y_mle']].values
ax.scatter(pixel_size*theta_mle[:,1],pixel_size*theta_mle[:,0],color='red',s=5,label='MLE')
ax.scatter(pixel_size*theta_cvdm[:,1],pixel_size*theta_cvdm[:,0],color='blue',s=5,label='CVDM')
ax.set_xlabel(r'$\theta_{u} \;[um]$',fontsize=16)
ax.set_ylabel(r'$\theta_{v} \;[um]$',fontsize=16)
ax.set_aspect(1.0)
ax.set_xticks([0.1,0.5,1.0,1.5])
ax.set_yticks([0.1,0.5,1.0,1.5])
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=2,frameon=False,fontsize=12)
plt.tight_layout()
plt.savefig('/N/slate/cwseitz/cvdm/Sim/4x/Figure-8.png',dpi=300)
plt.show()

#inter,union,fp,fn
errordf,set_metrics = get_errors(spots_cvdm,theta_mle.T,upsample=1,tol=2.0)
print(theta_mle.shape,theta_cvdm.shape)
print(np.sum(set_metrics,axis=0)[0])


