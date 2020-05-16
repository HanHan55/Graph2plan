# Graph2plan
An interactive interface, where the user can edit the retrieved graph and adapt it as needed. The user can add or delete room nodes and/or adjacency edges, or move nodes around to change the layout.

![Interface Image](https://github.com/HanHan55/Graph2plan/raw/master/Interface/Img/interface.jpg)

#### Graph2Plan: Learning Floorplan Generation from Layout Graphs
[project page](https://vcc.tech/research/2020/Graph2Plan) | [video & demo](https://vcc.tech/file/upload_file//image/research/att202005081043/Graph2Plan.mp4)   

Ruizhen Hu, Zeyu Huang, Yuhan Tang, [Oliver Van kaick](http://people.scs.carleton.ca/~olivervankaick/index.html), [Hao Zhang](http://www.cs.sfu.ca/~haoz/), [Hui Huang](http://vcc.szu.edu.cn/index-2.html)

SIGGRAPH 2020

We introduce a learning framework for automated floorplan generation which combines generative modeling using deep neural networks and userin- the-loop designs to enable human users to provide sparse design constraints. Such constraints are represented by a layout graph. The core component of our learning framework is a deep neural network, Graph2Plan, which converts a layout graph, along with a building boundary, into a floorplan that fulfills both the layout and boundary constraints. Given an input building boundary, we allow a user to specify room counts and other layout constraints, which are used to retrieve a set of floorplans, with their associated layout graphs, from a database. For each retrieved layout graph, along with the input boundary, Graph2Plan first generates a corresponding raster floorplan image, and then a refined set of boxes representing the rooms. Graph2Plan is trained on RPLAN, a large-scale dataset consisting of 80K annotated floorplans. The network is mainly based on convolutional processing over both the layout graph, via a graph neural network (GNN), and the input building boundary, as well as the raster floorplan images, via conventional image convolution. We demonstrate the quality and versatility of our floorplan generation framework in terms of its ability to cater to different user inputs.We conduct both qualitative and quantitative evaluations, ablation studies, and comparisons with state-of-the-art approaches.

![Paper Image](https://github.com/HanHan55/Graph2plan/raw/master/Interface/Img/paper.png)


#### Contact
If you have any questions or find any bugs, please contact us: Ruizhen Hu(ruizhen.hu@gmail.com), Zeyu Huang(vcchzy@gmail.com), Yuhan Tang(yuhantang55@gmail.com)

# Interface
### Installation
This implementation requires the following dependencies (tested on Window 10):

* Conda

* Python 3.7 

		conda create -n Graph2Plan python==3.7
		conda activate Graph2Plan
	
* SciPy, Django, Torch, Opencv, Pandas. You can quickly install/update these dependencies by running the following (replace pip with pip3 for Python 3.7):

		conda install scipy django  pytorch==1.3.1 opencv torchvision==0.4.2 pandas==0.25.3
		
* The matlab engine:
  Find the file "setup.py" in <path_to_Matlab>\<version>\extern\engines\python\"
  
		python setup.py install
		


###  Run

* Download Data:

	http://vcc.tech/file/upload_file/Data/G2P/Data.7z		
	
	unzip the data folder
	
	Place the downloaded data files in the folder
	
		'Data\Interface\House\retrieval\' to 'Interface\retrieval\'
		'Data\Interface\House\static\' to 'Interface\static\'
		'Data\Network\' to 'Network\'
* Run project:

		python manage.py runserver 0.0.0.0:8000
* Open browser:
			
	http://127.0.0.1:8000/home

# Network
### Requirements

 The same vertion of Scipy, Django, Torch, Opencv, Pandas with the interface

```
pip install tqdm tensorboardX pytorch-ignite==0.2.1
```

### Prepare data

**Data format**

```
data.mat
    - name: file name in RPLAN dataset
    - boundary: (x,y,dir,isNew), first two point indicate the front door
    - order: room order for visualization
    - rType: room categories
    - rBoundary: (x,y), boundary points for each room
    - gtBox: (y0,x0,y1,x1), left-top(in) and right-bottom(out) corner of a room, extracted directly from the RPLAN dataset
    - gtBoxNew: (x0,y0,x1,y1), extracted after filling the gap between rooms
    - rEdge: (u,v,r), room indices and relative position(u relative to v)
```

**Split data for training**

```
python split.py
```

### Train

```
python train.py
```

### Application

Copy the checkpoint to the web application directory `Interface\model` and rename it as `model.pth`

### Acknowledgement

Some source files are borrowed from:
- sg2im: https://github.com/google/sg2im
- scene generation: https://github.com/ashual/scene_generation

Data are extracted from http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
