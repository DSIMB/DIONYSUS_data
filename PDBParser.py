import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist, squareform
from components import covalent_radii, component_to_type, carbohydrates, carbohydrates_to_ring
import numpy as np
import os
import networkx as nx

class LightPDBParser:
    cutoff = 3
    factor_covalent = 1.1
    dry = True
    remove_hydrogens = True
    remove_extra = True
    def __init__(self, pdb_path, alt_loc=None, include_carb=True, model=1, chains=None):
        self.pdb_path = pdb_path
        self.alt_loc = alt_loc
        self.include_carb = include_carb
        self.chains = chains
        atomic_coordinates = pd.read_fwf(pdb_path, 
                            colspecs=[(0, 6), (6, 11), (12, 16), (16, 17), (17, 21), 
                                    (21, 22), (22, 26), (26, 28), (30, 38), 
                                    (38, 46), (46, 54), (54, 60), (60, 66), 
                                    (72, 76), (76, 78), (78, 80)])
        atomic_coordinates.columns = [
            'record_type', 'atom_id', 'atom_name', 'alt_loc', 
            'residue_name', 'chain_id', 'residue_seq_id', 
            'code_ins_residue', 'cart_x', 'cart_y', 'cart_z', 
            'occupancy', 't_factor', 'seg_id', 'element', 'charge']


        models_info = (atomic_coordinates.query('record_type.str.startswith("MODEL")')).copy()
        self.conect_info = os.popen(f"grep ^CONECT {pdb_path}")
        # self.conect_info = pd.read_csv(pdb_path)('record_type.str.startswith("CONECT")'))
        if not models_info.empty:
            self.n_models = len(models_info)
            if model is not None:
                if models_info['atom_name'].dtype == object:
                    next_model = str(model+1)
                    model = str(model)
                else:
                    next_model = model+1
                beg_ix = models_info.query('atom_name==@model').index[0]
                end_ix = models_info.query('atom_name==@next_model').index[0]
                atomic_coordinates = atomic_coordinates.iloc[beg_ix:end_ix].copy()
                atomic_coordinates['model'] = model
            else:
                models_info['atom_name'] = models_info['atom_name'].astype(int)
                for m in models_info.atom_name.unique(): 
                    beg_ix = models_info.query('atom_name==@m').index
                    end_ix = models_info.query('atom_name==(@m+1)').index
                    if not end_ix.empty:
                        atomic_coordinates.loc[beg_ix[0]:end_ix[0], 'model'] = m
                    else:
                        atomic_coordinates.loc[beg_ix[0]:, 'model'] = m
        else:
            self.n_models = 1
            atomic_coordinates['model'] = 1
        atomic_coordinates = atomic_coordinates.query('record_type.isin(["ATOM", "HETATM"])').copy()
        atomic_coordinates['model'] = atomic_coordinates['model'].astype(int).astype(str)
        
        # Dealing with insertion codes
        try:
            atomic_coordinates['residue_seq_id'] = atomic_coordinates['residue_seq_id'].astype(int).astype(str)
        except:
            atomic_coordinates[['tmp_residue_seq_id', 'tmp_code_ins_residue']] = (
                atomic_coordinates['residue_seq_id'].str.extract('(\d+)([A-Za-z]+)', 
                                                                 expand=True))
            atomic_coordinates['tmp_residue_seq_id'].fillna(atomic_coordinates['residue_seq_id'],
                                                            inplace=True)
            atomic_coordinates['tmp_code_ins_residue'].fillna(atomic_coordinates['code_ins_residue'],
                                                              inplace=True)
            atomic_coordinates = atomic_coordinates.drop(['residue_seq_id', 'code_ins_residue'], axis=1)\
                                                   .rename({
                'tmp_residue_seq_id': 'residue_seq_id',
                'tmp_code_ins_residue': 'code_ins_residue'},
                axis=1)

        atomic_coordinates = atomic_coordinates.copy()
        # atomic_coordinates['element'] = atomic_coordinates['element'].fillna(atomic_coordinates['atom_id'].str[0])
        atomic_coordinates['covalent_radii'] = atomic_coordinates['element'].astype(str).str.title().map(covalent_radii)
        atomic_coordinates['residue_type'] = atomic_coordinates['residue_name'].map(component_to_type)
        atomic_coordinates['node_label'] = atomic_coordinates['residue_type'].str.upper().str[0]+':'+\
                                           atomic_coordinates['residue_name'].fillna('').astype(str)+ ':'+ \
                                           atomic_coordinates['residue_seq_id'].fillna('').astype(str)+':'+ \
                                           atomic_coordinates['code_ins_residue'].fillna('').astype(str)+':'+\
                                           atomic_coordinates['chain_id'].astype(str).fillna('') 
        atomic_coordinates['atom_id'] = atomic_coordinates['atom_id'].astype(str)
        self.atomic_coordinates = atomic_coordinates.reset_index(drop=True).fillna('').copy()
        self.has_carbohydrates = self.atomic_coordinates.residue_name.isin(carbohydrates).any()
        self._clean_structure()
        self.original_coordinates = atomic_coordinates.copy()
        # self.atomic_coordinates['id'] = self.atomic_coordinates.index
        # self._merge_binding_sites()
        # self._get_components_type()
    
    def _clean_structure(self):
        cond = []
        if self.remove_hydrogens:
            cond.append('(element !="H")')
        if self.dry:
            cond.append('(residue_name!="HOH")')
        if self.remove_extra:
            if not self.include_carb:
                cond.append('(residue_type=="amino_acid")')
            else:
                cond.append('(residue_type.isin(["amino_acid", "carbohydrate"]))')
        if self.alt_loc is not None:
            if self.alt_loc=="none":
                cond.append('(alt_loc=="")')
            else:
                cond.append('(alt_loc.isin(["", @self.alt_loc]))')
        if self.chains is not None:
            cond.append('(chain_id.isin(@self.chains))')
        if cond:
            self.atomic_coordinates = self.atomic_coordinates.query(
                ' & '.join(cond))
    
    def _get_sasa(self):
        """ Method computing the solvent accessibility surface area by atom"""
        import freesasa as fs
        classifier = fs.Classifier()
        radii, polarity = zip(*[(classifier.radius(resn, atomn), classifier.classify(resn, atomn))
                for (resn, atomn) in self.atomic_coordinates[['residue_name', 'atom_name']].values])                                            
        self.atomic_coordinates['free_sasa_radius'] = radii
        # self.atomic_coordinates['polarity'] = pd.Series(polarity).map({'Apolar': False,
                                                                    #    'Polar': True})
        alt_locations = self.atomic_coordinates['alt_loc'].unique().tolist()
        for model in self.atomic_coordinates['model'].unique():
            for alt_location in alt_locations:
                comput_loc = ((self.atomic_coordinates['free_sasa_radius']>0)
                            & (self.atomic_coordinates['model']==model)
                            & (self.atomic_coordinates['alt_loc'].isin([alt_location, ''])))
                coordinates = self.atomic_coordinates.loc[comput_loc,
                                [f'cart_{axis}' for axis in ['x', 'y', 'z']]].values.astype(float)
                radii = self.atomic_coordinates.loc[comput_loc, 'free_sasa_radius'].values
                result = fs.calcCoord(coordinates.flatten(), radii)
                self.atomic_coordinates.loc[comput_loc,
                                        f'sasa_{alt_location}'] = [result.atomArea(i) for i in range(comput_loc.sum())]
        self.atomic_coordinates['sasa'] = self.atomic_coordinates[[f'sasa_{alt_location}' for alt_location in alt_locations]].mean(axis=1)
        self.sasa = self.atomic_coordinates.groupby(['model',
                                                     'residue_seq_id',
                                                     'residue_name',
                                                     'chain_id',
                                                     'code_ins_residue',
                                                     ], sort=False)['sasa'].sum()        

    def extract_chain_info(self):
        chains = []
        chain_lengths = []
        chain_types = []
        for chain, chain_data in self.atomic_coordinates.query("model==1")\
                                                        .groupby('chain_id'):
            chains.append(chain)
            residue_list = chain_data.groupby(['residue_name', 'residue_seq_id', 'code_ins_residue'])\
                                     .first()\
                                     .reset_index()
            chain_length = len(residue_list)
            chain_lengths.append(chain_length)
            components_types = residue_list['residue_name'].map(component_to_type)
            if (components_types == 'amino_acid').sum() > 30:
                chain_type = 'Protein'
            elif (components_types == 'amino_acid').sum() >= chain_length/2:
                chain_type = 'Peptide'
            elif (components_types == "carbohydrate").sum() >= chain_length/2:
                chain_type = 'Saccharide'
            elif (components_types == "nucleic_acid").sum() >= chain_length/2:
                chain_type = "Nucleic acid"
            else:
                chain_type = "Other"
            chain_types.append(chain_type)

        # Distinguish glycosylations from free ligands
        chain2type = dict(zip(chains, chain_types))
        protein_chains = [chain for chain, chaintype in chain2type.items() if chaintype=="Protein"]
        best_coordinates = self.atomic_coordinates.query("(model==1)")\
                                                    .sort_values('occupancy', ascending=False)\
                                                    .reset_index(names='index')\
                                                    .groupby(['atom_name',
                                                              'residue_seq_id',
                                                              'chain_id',
                                                              'code_ins_residue'],
                                                            sort=False)\
                                                    .first()\
                                                    .sort_values('index')
        protein_coordinates = best_coordinates.query('chain_id.isin(@protein_chains)').copy()
        covalent_radii_prot = protein_coordinates['element'].map(covalent_radii).values

        annotated_covalent_bonds = [(line.split()[1], v)
                           for line in self.conect_info.readlines()
                           for v in line.split()[2:]
                           if int(line.split()[1]) < int(v)]
        atom_id2chain = dict(zip(self.atomic_coordinates['atom_id'],
                                 self.atomic_coordinates['chain_id']))
        chain_covalent_bonds = [(atom_id2chain[u], atom_id2chain[v])
                                for u, v in annotated_covalent_bonds
                                if (u in atom_id2chain and v in atom_id2chain) and 
                                   (atom_id2chain[u] != atom_id2chain[v])]
                    
        for chain, chaintype in chain2type.items():
            if chaintype != "Saccharide":
                continue
            is_glyco = False
            # Pass through annotated glycosylations first (should be faster)
            for u, v in chain_covalent_bonds:
                if chain==u:
                    is_glyco = (chain2type[v] == "Protein")
                    if is_glyco:
                        break
                elif chain==v:
                    is_glyco = (chain2type[u] == "Protein")
                    if is_glyco:
                        break
            if is_glyco:
                chain2type[chain] = 'Covalent carbohydrate'
                continue
            # Then computes glycosylations using atomic coordinates
            carb_coords = self.atomic_coordinates.query('(chain_id==@chain) & (residue_type=="carbohydrate")').copy()
            distances = cdist(*(list(map(lambda X: X[['cart_x', 'cart_y', 'cart_z']].astype(float).values, [carb_coords, protein_coordinates]))))
            covalent_radii_carb = carb_coords['element'].map(covalent_radii).values
            ideal_bond_length = np.add.outer(covalent_radii_carb, covalent_radii_prot)
            if np.any(distances < 1.1*ideal_bond_length):
                chain2type[chain] = 'Covalent carbohydrate'
            else:
                chain2type[chain] = 'Free carbohydrate'

        chain_info = pd.DataFrame({'Id': chains,
                             'Length': chain_lengths})
        chain_info['Type'] = chain_info['Id'].map(chain2type)
        return chain_info

    def extract_binding_site_info(self, only_surface_atoms: bool=True, add_atoms=['CA']):
        if isinstance(add_atoms, str):
            add_atoms = [add_atoms]
        if only_surface_atoms and not hasattr(self, 'sasa'):
            self._get_sasa()

        # Get carbohydrate coordinates
        carbohydrates_coordinates = self.atomic_coordinates.query('residue_type=="carbohydrate"')\
                                                           .drop_duplicates(subset=['model', 
                                                              'residue_name',
                                                              'residue_seq_id',
                                                              'chain_id',
                                                              'alt_loc',
                                                              'code_ins_residue'])
        # for (model, residue_name, ), residue_data \
            # in carbohydrates.groupby(
                # ['model', 'residue_name', 'residue_seq_id', 'chain_id', 'alt_loc', 'code_ins_residue']):
        carbohydrates_coordinates['rings'] = carbohydrates_coordinates['residue_name'].map(
            lambda X: carbohydrates_to_ring[X]
        )
        # Checking if the dataframe gets bigger when exploding rings (i.e. several rings in one residue)
        len_before_ring_explode = len(carbohydrates_coordinates)
        carbohydrates_coordinates = carbohydrates_coordinates.explode('rings')
        # If some carbohydrates have several rings => show the column 
        visible_rings =  (len(carbohydrates_coordinates) != len_before_ring_explode)
        carbohydrates_coordinates['rings'] = carbohydrates_coordinates['rings'].map(
            lambda X: ','.join(X)
        )
        carbohydrates_coordinates['residue_fullname'] = carbohydrates_coordinates['residue_name']\
                                                        + carbohydrates_coordinates['residue_seq_id']\
                                                        + carbohydrates_coordinates['code_ins_residue']\
        # If some carbohydrates have alternate location we reduce to e.g. 'A'=''+'A' / 'B'=''+'B'
        alt_locs = carbohydrates_coordinates.query('alt_loc!=""')
        visible_alt_loc = not alt_locs.empty
        if not alt_locs.empty:
            carbohydrates_coordinates = carbohydrates_coordinates.query("~((node_label.isin(@alt_locs.node_label)) & (alt_loc==''))")

        # Get protein residues
        protein_condition = '(residue_type=="amino_acid")'
        if only_surface_atoms:
            protein_condition += ' & (sasa>0)'
        protein = self.atomic_coordinates.query(protein_condition)
        protein_coordinates = protein[['cart_x', 'cart_y', 'cart_z']].values.astype(float)
        
        # Iterate over all potential binding site to see if the protein contains more binding sites due to alternate locations
        all_binding_sites = []
        protein_alt_locations = []

        for i, (_, bs) in enumerate(carbohydrates_coordinates.iterrows(), start=1):
            m, resl, altloc, ring = bs[['model', 
                                            'node_label',
                                            'alt_loc',
                                            'rings']]
            ring = ring.split(',')
            carbohydrate_ring = self.atomic_coordinates.query(
                "(model==@m) & (node_label==@resl) & (alt_loc.isin(['', @altloc])) & (atom_name.isin(@ring))"
            ).copy()
            carbohydrate_ring_coordinates = carbohydrate_ring[['cart_x', 'cart_y', 'cart_z']].values.astype(float)
            distances = cdist(carbohydrate_ring_coordinates, protein_coordinates)
            if len(distances) == 0:
                row = np.array([])
            else:
                row = np.where(distances.min(0) < 7)
            binding_site = protein.iloc[row].query('model==@m')
            if not binding_site.empty:
                # Add atoms to better capture the protein structure (e.g. CA)
                multi_index = binding_site.groupby(['node_label', 'alt_loc']).first().index
                reslabel, altloc = zip(*multi_index) 
                atoms_to_add = self.atomic_coordinates.query(
                    "(node_label.isin(@reslabel))  & (alt_loc.isin(['', @altloc])) & (atom_name.isin(@add_atoms)) & (model==@m)"
                )
                binding_site = pd.concat([binding_site, atoms_to_add]).drop_duplicates()
            
            bs_alt_locations = binding_site['alt_loc'].unique()

            if len(bs_alt_locations) > 1:
                # Reducing alt locations
                alt_loc_reduced = [elt for elt in bs_alt_locations if elt != '']
                protein_alt_locations.append(list(alt_loc_reduced))
                for loc in alt_loc_reduced:
                    all_binding_sites.append(pd.concat([carbohydrate_ring, binding_site.query('alt_loc.isin(["", @loc])')]))
            else:
                protein_alt_locations.append([''])
                all_binding_sites.append(pd.concat([carbohydrate_ring, binding_site]))

        carbohydrates_coordinates['prot_alt_loc'] = protein_alt_locations

        self.binding_sites = carbohydrates_coordinates.explode('prot_alt_loc')#\
                                                    #   .sort_values(['model', 
                                                    #                 'chain_id', 
                                                    #                 'residue_seq_id', 
                                                    #                 'code_ins_residue', 
                                                    #                 'alt_loc', 
                                                    #                 'prot_alt_loc'])
        visible_prot_alt_loc = len(carbohydrates_coordinates) != len(self.binding_sites)

        coordinates_backup = self.atomic_coordinates.copy()

        for i, bs in enumerate(all_binding_sites, start=1):
            self.atomic_coordinates = bs
            out_path = self.pdb_path.replace('.pdb', f'_{i}.pdb')
            out_path_vizu = self.pdb_path.replace('.pdb', f'_{i}_vizu.pdb')
            self.write_for_patchsearch(out_path)
            self.write_for_patchsearch(out_path_vizu, output_type="pdb", fill_carbs=True)
        self.atomic_coordinates = coordinates_backup.copy()

        visible_models = self.n_models > 1

        return self.binding_sites, visible_rings, visible_alt_loc, visible_prot_alt_loc, visible_models
    
    def merge_binding_sites(self):
        if not hasattr(self, "binding_sites"):
            self.extract_binding_site_info()
        binding_sites = self.binding_sites.reset_index()
        polysaccharides_indexes = []
        # Not merging binding sites from different models/alt locations
        for model in self.binding_sites.model.unique():
            for alt_loc in self.binding_sites.alt_loc.unique():
                # Select carbohydrates
                carbohydrates = self.atomic_coordinates.query("(alt_loc.isin(['', @alt_loc])) & (model==@model) & (node_label.isin(@self.binding_sites.node_label.values))""")
                carbohydrate_coordinates = carbohydrates[['cart_x', 'cart_y', 'cart_z']].values.astype(float)
                # Compute distances, and ideal covalent distances
                distances = squareform(pdist(carbohydrate_coordinates))
                cov_rads = carbohydrates.covalent_radii.values
                ideal_bond_length = np.add.outer(cov_rads, cov_rads)
                # Compare effective distance between atoms and the ideal covalent distance
                row, col = np.where(distances<=ideal_bond_length*1.1)
                ix = row < col
                row, col = row[ix], col[ix]
                # Get node labels for atoms covalently bound 
                node_labels = carbohydrates.node_label.values
                row_res, col_res = node_labels[row], node_labels[col]
                ix = np.where(row_res != col_res)
                row_res, col_res = row_res[ix], col_res[ix]
                edges = list(zip(row_res, col_res))
                G = nx.Graph()
                G.add_edges_from(edges)
                G.add_nodes_from(self.binding_sites.node_label.unique())
                for polysaccharide in nx.connected_components(G):
                    prot_alt_locs = binding_sites.prot_alt_loc.unique()
                    polysaccharides_ix = binding_sites.query("(node_label.isin(@polysaccharide)) & (model==@model) & (alt_loc.isin(['', @alt_loc]))").index.tolist()
                    if len(prot_alt_locs) == 1:
                        if len(polysaccharides_ix) > 1:
                            polysaccharides_indexes.append(polysaccharides_ix)
                        continue
                    for prot_alt_loc in binding_sites.prot_alt_loc.unique():
                        if prot_alt_loc == '':
                            continue
                        if len(polysaccharides_ix) > 1:
                            polysaccharides_indexes.append(binding_sites.query("(node_label.isin(@polysaccharide)) & (model==@model) & (alt_loc.isin(['', @alt_loc])) & (prot_alt_loc.isin(['', @prot_alt_loc]))").index.tolist())

        for i, polysacch in enumerate(polysaccharides_indexes, start=1):
            concatenated_data = pd.DataFrame()
            for j in polysacch:
                file_path = self.pdb_path.replace('.pdb', f'_{j+1}_vizu.pdb')
                data = pd.read_fwf(file_path, widths=[81], names=['lines'])
                concatenated_data = pd.concat([concatenated_data, data]).drop_duplicates()
            output_path = self.pdb_path.replace('.pdb', f'_poly_{i}_vizu.pdb')
            with open(output_path, 'w') as output_file:
                if 'lines' in concatenated_data.columns:
                    for line in concatenated_data.lines.values:
                        output_file.write(line+'\n')
            
        return polysaccharides_indexes
        
    def get_atomic_contacts(self, cutoff: float=5):
        coordinates = self.atomic_coordinates.query('residue_type=="amino_acid"')\
                                             .sort_values('occupancy', ascending=False)\
                                             .reset_index(names='index')\
                                             .groupby(['atom_name',
                                                      'residue_seq_id',
                                                      'chain_id',
                                                      'code_ins_residue'],
                                                      sort=False)\
                                              .first()\
                                              .sort_values('index')\
                                              [['cart_x', 'cart_y', 'cart_z']].values
        self.polymer_residues = self.atomic_coordinates.query('residue_type=="amino_acid"')['node_label'].unique()
        tree = KDTree(coordinates)
        distance_matrix = tree.sparse_distance_matrix(tree, cutoff, output_type='coo_matrix').tocsr()
        row, col = distance_matrix.nonzero()
        ix = row < col
        row, col = row[ix], col[ix]
        distances = np.array(distance_matrix[(row, col)]).squeeze()

        atomic_contacts = pd.DataFrame({'id': row.astype(str),
                                        'id2': col.astype(str),
                                        'distance': distances})

        atomic_contacts = atomic_contacts.merge(
            self.atomic_coordinates.rename(columns={'atom_id': 'id'}), on='id', how='inner', suffixes=(None, '1')
                                        ).rename(
            columns={'id': 'id1', 'id2': 'id'}
                                        ).merge(
            self.atomic_coordinates.rename(columns={'atom_id': 'id'}), on='id', how='inner', suffixes=('1', '2')
                                        ).rename(columns={'id': 'id2'})
        
        atomic_contacts = atomic_contacts.query('(alt_loc1 == alt_loc2) | (alt_loc1.isna()) | (alt_loc1.isna())')
        self.atomic_contacts = atomic_contacts
    
    def get_residue_contacts(self, remove_intraresidual: bool=True, 
                             keep_closest: int=0, **kwargs):
        if not hasattr(self, 'atomic_contacts'):
            self.get_atomic_contacts(**kwargs)
        residue_contacts = pd.DataFrame((self.atomic_contacts.value_counts(['node_label1', 'node_label2'], sort=False)))
        residue_contacts['min_distance'] = self.atomic_contacts.sort_values('distance')\
                                                            .groupby(['node_label1', 'node_label2'], sort=False)\
                                                            .first()['distance']
        residue_contacts = residue_contacts.rename({0: 'count'}, axis=1).reset_index()
        if remove_intraresidual:
            residue_contacts = residue_contacts.query('node_label1!=node_label2').copy()
        self.label_id_to_index = dict(zip(self.polymer_residues, range(len(self.polymer_residues))))
        residue_contacts['resid_1'] = residue_contacts['node_label1'].map(self.label_id_to_index)
        residue_contacts['resid_2'] = residue_contacts['node_label2'].map(self.label_id_to_index)
        self.graph = residue_contacts[['resid_1', 
                                       'resid_2',
                                       'count',
                                       'min_distance']]\
                                       .sort_values(['resid_1', 'resid_2'])\
                                       .reset_index(drop=True)
        if keep_closest:
            networkx_graph = nx.from_pandas_edgelist(self.graph, 
                                                     source="resid_1",
                                                     target="resid_2",
                                                     edge_attr=["count", "min_distance"])
            edge_distances = nx.get_edge_attributes(networkx_graph, "min_distance")
            edge_distances.update({(v, u): value for (u, v), value in edge_distances.items()})
            edges_to_remove = [sorted([(u, v) for v in nx.neighbors(networkx_graph, u)],
                               key=lambda uv : edge_distances[(uv[0], uv[1])])[keep_closest:]
                               for u in networkx_graph.nodes()]
            edges_to_remove = [edge for l in edges_to_remove for edge in l]
            networkx_graph.remove_edges_from(edges_to_remove)
            self.graph = nx.to_pandas_edgelist(networkx_graph, source="resid_1", target="resid_2")

    def get_graph(self, **kwargs):
        """ Method transforming the raw graphs in torch data"""
        if not hasattr(self, 'graph'):
            self.get_residue_contacts(**kwargs)
        edge_index = torch.as_tensor(self.graph[['resid_1', 'resid_2']].values, dtype=torch.long) 
        edge_attr = torch.as_tensor(self.graph[['count', 'min_distance']].values, dtype=torch.float)
        return Data(edge_index=edge_index.t().contiguous(), 
                    counts=edge_attr[:,0],
                    distances=edge_attr[:,1],
                    node_labels=self.polymer_residues)
  

    def _get_covalent_linkage(self, selection="all"):
        if selection == "carbohydrates":
            df = self.atomic_coordinates.query("residue_type=='carbohydrate'")
        else:
            df = self.atomic_coordinates
        coordinates = df[['cart_x', 'cart_y', 'cart_z']].values
        tree = KDTree(coordinates)
        distance_matrix = tree.sparse_distance_matrix(tree, self.cutoff, output_type='coo_matrix').tocsr()
        row, col = distance_matrix.nonzero()
        ix = row < col
        row, col = row[ix], col[ix]
        row, col = df.index[row].values, df.index[col].values
        distances = np.array(distance_matrix[(row, col)]).squeeze()
        atomic_contacts = pd.DataFrame({'id': row,
                                        'id2': col,
                                        'distance': distances})

        atomic_contacts = atomic_contacts.merge(
            self.atomic_coordinates, on='id', how='inner', suffixes=(None, '1')
                                        ).rename(
            columns={'id': 'id1', 'id2': 'id'}
                                        ).merge(
            self.atomic_coordinates, on='id', how='inner', suffixes=('1', '2')
                                        ).rename(columns={'id': 'id2'})
        
        atomic_contacts = atomic_contacts.query('(alt_loc1 == alt_loc2) | (alt_loc1.isna()) | (alt_loc1.isna())')
        atomic_contacts['ideal_bond_length'] = atomic_contacts['covalent_radii1'] + atomic_contacts['covalent_radii2']
        
        atomic_contacts['is_covalent'] = (
            atomic_contacts['ideal_bond_length']*self.factor_covalent >
            atomic_contacts['distance'])

        contacts_to_add = [(line.split()[1], v)
                           for line in self.conect_info.readlines()
                           for v in line.split()[2:]
                           if int(line.split()[1]) < int(v)]

        row, col = zip(*contacts_to_add)
        conect_covalent = pd.DataFrame({'atom_id': row,
                                        'atom_id2': col,
                                        'distance': np.inf,
                                        'is_covalent': True})
        conect_covalent = conect_covalent.merge(
            self.atomic_coordinates, on='atom_id', how='inner', suffixes=(None, '1')
                                        ).rename(
            columns={'atom_id': 'atom_id1', 'atom_id2': 'atom_id'}
                                        ).merge(
            self.atomic_coordinates, on='atom_id', how='inner', suffixes=('1', '2')
                                        ).rename(columns={'atom_id': 'atom_id2'})
        atomic_contacts = pd.concat([atomic_contacts, conect_covalent])\
                                   .sort_values(by=['is_covalent', 'distance'],
                                                ascending=[False, True])\
                                   .drop_duplicates(
                                    ['atom_id1', 'atom_id2', 'id1', 'id2'])
        covalent_linkage = nx.Graph()
        edges = atomic_contacts.query('is_covalent')[['node_label1', 'node_label2']].values
        covalent_linkage.add_edges_from(edges)
        self.covalent_linkage = covalent_linkage

    def _get_components_type(self):
        for component in nx.connected_components(self.covalent_linkage):
            n_aa = sum([comp[0]=='A' for comp in component])
            # A component without amino acid needs no change
            if not n_aa:
                continue
            n_carb = sum([comp[0]=='C' for comp in component])
            # Simple protein
            if n_aa >= 30 and not n_carb:
                continue
            # Peptide should be counted as other 
            if n_aa < 30 and not n_carb:
                pass
    
    def write_for_patchsearch(self, output_path, output_type="patchsearch", fill_carbs=False):

        def whitespace_gen_left(values, expected_size, char=' '):
            return values.map(lambda X: X[:expected_size]) + [char*size for size in (expected_size-values.str.len())]
        def whitespace_gen_right(values, expected_size, char=' '):
            return [char*size for size in (expected_size-values.str.len())] + values.map(lambda X: X[:expected_size])

        def get_line(bs):
            def gen_coordinates(string):
                value = bs[f'cart_{string}'].astype(str)
                # return ['' if elt else ' ' for elt in negative_coordinates]\
                #     + whitespace_gen_left(value, 6, ' ')\
                #     + [' ' if elt else '' for elt in negative_coordinates]
                return value.map(lambda X: '' if X.startswith('-') else ' ')\
                     + whitespace_gen_left(value, 7, ' ')\
                     + value.map(lambda X: ' ' if X.startswith('-') else '')\


            base_dic = {'CA': 'A', 'O': 'O', 'C': 'C', 'N': 'N', 'CB': 'b', 'C1': 's'}
            def get_patchsearch_typing(resname, element, entity_type):
                if entity_type=="carbohydrate":
                    if element=="C":
                        return "g"
                    else:
                        return "u"
                if resname in ['TYR', 'PHE', 'TRP', 'HIS'] and element=='C':
                    return 'a'
                if element in ['O', 'N', 'C']:
                    return element.lower()
                else:
                    return element
            
            reverse_type = {"A": "C", "b": "C", "s": "C", "g": "C", "u": "O", "a": "C"}
            
            if output_type=="patchsearch":
                # if "sasa" in bs.columns:
                #     bs = bs.query('sasa>0')
                bs['patchsearch_type'] = [
                    base_dic[atomn] if atomn in base_dic else get_patchsearch_typing(resn, elem, entity_type)
                    for (atomn, resn, elem, entity_type) in 
                    bs[['atom_name', 'residue_name', 'element', 'residue_type']].values]
                bs['new_chain'] = [' A' if elt=='amino_acid' else ' B' for elt in (bs['residue_type'])]
            else:
                bs['patchsearch_type'] = bs['element'].map(
                    lambda X: reverse_type.get(X, X.upper())
                )
                bs['new_chain'] = ' '+ bs['chain_id']
            
            #bs['alt_loc'] = bs['alt_loc'].map(lambda X: {'': ' '}.get(X, X)) 
            # Getting rid of alt locations for the sake of simplicity
            bs['alt_loc'] = ' '

            bs['line'] = whitespace_gen_left(bs['record_type'], 6)  \
                    + whitespace_gen_right(bs['atom_id'].astype(float).astype(int).astype(str), 5) \
                    + ' ' \
                    + [' '+atom+(3-len(atom))*' ' if c1&c2 else ''+atom+(4-len(atom))*' ' for (c1, c2, atom) in zip(bs['element'].str.len()==1, bs['atom_name'].str.len()<4, bs['atom_name'])]\
                    + bs['alt_loc'].fillna(' ').astype(str) \
                    + whitespace_gen_right(bs['residue_name'], 3) \
                    + bs['new_chain'] \
                    + whitespace_gen_right(bs['residue_seq_id'].astype(int).astype(str), 4) \
                    + whitespace_gen_right(bs['code_ins_residue'].fillna(' '), 1) \
                    + ' '*3 \
                    + gen_coordinates('x')\
                    + gen_coordinates('y')\
                    + gen_coordinates('z')\
                    + whitespace_gen_right(bs['occupancy'].astype(str), 6) \
                    + whitespace_gen_right(bs['t_factor'].astype(str), 6) \
                    + ' '*10 \
                    + bs['patchsearch_type'] \
                    + '   '
                    # + bs['charge'].fillna(' ')
            return bs
        if fill_carbs:
            carbs = self.atomic_coordinates.query('residue_type=="carbohydrate"')
            label_carbs = carbs.node_label
            filled_atoms = self.original_coordinates.query('node_label.isin(@label_carbs)')
            self.atomic_coordinates = pd.concat([self.atomic_coordinates,
                                                filled_atoms]).drop_duplicates(['atom_id'])
        lines = get_line(self.atomic_coordinates)
        lines['line'].to_csv(output_path, index=False)

if __name__ == '__main__':
    l = LightPDBParser('test/9DIP.pdb', model=None)
    aa = l.extract_binding_site_info(only_surface_atoms=True, add_atoms=['CA'])
