import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
import models.place_resnet as place_resnet
from models.pvtv2 import pvt_v2_b2


places365_classes = ['airfield', 'airplane_cabin', 'airport_terminal', 'alcove', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'apartment_building-outdoor', 'aquarium', 'aqueduct', 'arcade', 'arch', 'archaelogical_excavation', 'archive', 'arena-hockey', 'arena-performance', 'arena-rodeo', 'army_base', 'art_gallery', 'art_school', 'art_studio', 'artists_loft', 'assembly_line', 'athletic_field-outdoor', 'atrium-public', 'attic', 'auditorium', 'auto_factory', 'auto_showroom', 'badlands', 'bakery-shop', 'balcony-exterior', 'balcony-interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'bank_vault', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basketball_court-indoor', 'bathroom', 'bazaar-indoor', 'bazaar-outdoor', 'beach', 'beach_house', 'beauty_salon', 'bedchamber', 'bedroom', 'beer_garden', 'beer_hall', 'berth', 'biology_laboratory', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth-indoor', 'botanical_garden', 'bow_window-indoor', 'bowling_alley', 'boxing_ring', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'bus_station-indoor', 'butchers_shop', 'butte', 'cabin-outdoor', 'cafeteria', 'campsite', 'campus', 'canal-natural', 'canal-urban', 'candy_store', 'canyon', 'car_interior', 'carrousel', 'castle', 'catacomb', 'cemetery', 'chalet', 'chemistry_lab', 'childs_room', 'church-indoor', 'church-outdoor', 'classroom', 'clean_room', 'cliff', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'corn_field', 'corral', 'corridor', 'cottage', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk', 'dam', 'delicatessen', 'department_store', 'desert-sand', 'desert-vegetation', 'desert_road', 'diner-outdoor', 'dining_hall', 'dining_room', 'discotheque', 'doorway-outdoor', 'dorm_room', 'downtown', 'dressing_room', 'driveway', 'drugstore', 'elevator-door', 'elevator_lobby', 'elevator_shaft', 'embassy', 'engine_room', 'entrance_hall', 'escalator-indoor', 'excavation', 'fabric_store', 'farm', 'fastfood_restaurant', 'field-cultivated', 'field-wild', 'field_road', 'fire_escape', 'fire_station', 'fishpond', 'flea_market-indoor', 'florist_shop-indoor', 'food_court', 'football_field', 'forest-broadleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'garage-indoor', 'garage-outdoor', 'gas_station', 'gazebo-exterior', 'general_store-indoor', 'general_store-outdoor', 'gift_shop', 'glacier', 'golf_course', 'greenhouse-indoor', 'greenhouse-outdoor', 'grotto', 'gymnasium-indoor', 'hangar-indoor', 'hangar-outdoor', 'harbor', 'hardware_store', 'hayfield', 'heliport', 'highway', 'home_office', 'home_theater', 'hospital', 'hospital_room', 'hot_spring', 'hotel-outdoor', 'hotel_room', 'house', 'hunting_lodge-outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink-indoor', 'ice_skating_rink-outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn-outdoor', 'islet', 'jacuzzi-indoor', 'jail_cell', 'japanese_garden', 'jewelry_shop', 'junkyard', 'kasbah', 'kennel-outdoor', 'kindergarden_classroom', 'kitchen', 'lagoon', 'lake-natural', 'landfill', 'landing_deck', 'laundromat', 'lawn', 'lecture_room', 'legislative_chamber', 'library-indoor', 'library-outdoor', 'lighthouse', 'living_room', 'loading_dock', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market-indoor', 'market-outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'mezzanine', 'moat-water', 'mosque-outdoor', 'motel', 'mountain', 'mountain_path', 'mountain_snowy', 'movie_theater-indoor', 'museum-indoor', 'museum-outdoor', 'music_studio', 'natural_history_museum', 'nursery', 'nursing_home', 'oast_house', 'ocean', 'office', 'office_building', 'office_cubicles', 'oilrig', 'operating_room', 'orchard', 'orchestra_pit', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage-indoor', 'parking_garage-outdoor', 'parking_lot', 'pasture', 'patio', 'pavilion', 'pet_shop', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pier', 'pizzeria', 'playground', 'playroom', 'plaza', 'pond', 'porch', 'promenade', 'pub-indoor', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'repair_shop', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'river', 'rock_arch', 'roof_garden', 'rope_bridge', 'ruin', 'runway', 'sandbox', 'sauna', 'schoolhouse', 'science_museum', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall-indoor', 'shower', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'soccer_field', 'stable', 'stadium-baseball', 'stadium-football', 'stadium-soccer', 'stage-indoor', 'stage-outdoor', 'staircase', 'storage_room', 'street', 'subway_station-platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_hole', 'swimming_pool-indoor', 'swimming_pool-outdoor', 'synagogue-outdoor', 'television_room', 'television_studio', 'temple-asia', 'throne_room', 'ticket_booth', 'topiary_garden', 'tower', 'toyshop', 'train_interior', 'train_station-platform', 'tree_farm', 'tree_house', 'trench', 'tundra', 'underwater-ocean_deep', 'utility_room', 'valley', 'vegetable_garden', 'veterinarians_office', 'viaduct', 'village', 'vineyard', 'volcano', 'volleyball_court-outdoor', 'waiting_room', 'water_park', 'water_tower', 'waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'yard', 'youth_hostel', 'zen_garden']
mit67_classes = ['airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 'casino', 'children_room', 'church_inside', 'classroom', 'cloister', 'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice', 'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore', 'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden', 'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby', 'locker_room', 'mall', 'meeting_room', 'movietheater', 'museum', 'nursery', 'office', 'operating_room', 'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 'trainstation', 'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar']
sun397_classes = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'apartment_building/outdoor', 'apse/indoor', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival_gate/outdoor', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium', 'auto_factory', 'badlands', 'badminton_court/indoor', 'baggage_claim', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'basketball_court/outdoor', 'bathroom', 'batters_box', 'bayou', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'bistro/indoor', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden', 'bow_window/indoor', 'bow_window/outdoor', 'bowling_alley', 'boxing_ring', 'brewery/indoor', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'cabin/outdoor', 'cafeteria', 'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior/backseat', 'car_interior/frontseat', 'carrousel', 'casino/indoor', 'castle', 'catacomb', 'cathedral/indoor', 'cathedral/outdoor', 'cavern/indoor', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'chicken_coop/indoor', 'chicken_coop/outdoor', 'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'cloister/indoor', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'control_tower/outdoor', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'covered_bridge/exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle/office', 'dam', 'delicatessen', 'dentists_office', 'desert/sand', 'desert/vegetation', 'diner/indoor', 'diner/outdoor', 'dinette/home', 'dinette/vehicle', 'dining_car', 'dining_room', 'discotheque', 'dock', 'doorway/outdoor', 'dorm_room', 'driveway', 'driving_range/outdoor', 'drugstore', 'electrical_substation', 'elevator/door', 'elevator/interior', 'elevator_shaft', 'engine_room', 'escalator/indoor', 'excavation', 'factory/indoor', 'fairway', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'fire_escape', 'fire_station', 'firing_range/indoor', 'fishpond', 'florist_shop/indoor', 'food_court', 'forest/broadleaf', 'forest/needleleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'garage/indoor', 'garbage_dump', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'golf_course', 'greenhouse/indoor', 'greenhouse/outdoor', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'hot_tub/outdoor', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail/indoor', 'jail_cell', 'jewelry_shop', 'kasbah', 'kennel/indoor', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'labyrinth/outdoor', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'library/indoor', 'library/outdoor', 'lido_deck/outdoor', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'moat/water', 'monastery/outdoor', 'mosque/indoor', 'mosque/outdoor', 'motel', 'mountain', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'music_store', 'music_studio', 'nuclear_power_plant/outdoor', 'nursery', 'oast_house', 'observatory/outdoor', 'ocean', 'office', 'office_building', 'oil_refinery/outdoor', 'oilrig', 'operating_room', 'orchard', 'outhouse/outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor', 'parking_garage/outdoor', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pilothouse/indoor', 'planetarium/outdoor', 'playground', 'playroom', 'plaza', 'podium/indoor', 'podium/outdoor', 'pond', 'poolroom/establishment', 'poolroom/home', 'power_plant/outdoor', 'promenade_deck', 'pub/indoor', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'stadium/baseball', 'stadium/football', 'stage/indoor', 'staircase', 'street', 'subway_interior', 'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/indoor', 'synagogue/outdoor', 'television_studio', 'temple/east_asia', 'temple/south_asia', 'tennis_court/indoor', 'tennis_court/outdoor', 'tent/outdoor', 'theater/indoor_procenium', 'theater/indoor_seats', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'track/outdoor', 'train_railway', 'train_station/platform', 'tree_farm', 'tree_house', 'trench', 'underwater/coral_reef', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball_court/indoor', 'volleyball_court/outdoor', 'waiting_room', 'warehouse/indoor', 'water_tower', 'waterfall/block', 'waterfall/fan', 'waterfall/plunge', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'wine_cellar/barrel_storage', 'wine_cellar/bottle_storage', 'wrestling_ring/indoor', 'yard', 'youth_hostel']


class OSFA(nn.Module):
    def __init__(self, base_dim=2048, clip_dim=512, hidden_dim=512,
                 num_class=365, base_name='resnet50_places365', clip_name='ViT-B/32'):
        super(OSFA, self).__init__()

        # image encoder
        self.backbone = Backbone(num_class, base_name, clip_name)
        self.projection_cv = nn.Linear(clip_dim, hidden_dim)
        self.projection_ct = nn.Linear(clip_dim, hidden_dim)
        self.projection_bv = nn.Linear(base_dim, hidden_dim)
        self.projection_st_ou = nn.Linear(hidden_dim, clip_dim)
        self.projection_ct_ou = nn.Linear(hidden_dim, clip_dim)

        # image agg
        self.sfg = SceneFG(hidden_dim)
        self.olca = CrossAttnAgg(hidden_dim)
        # text agg
        self.cafg = CategoryAwareFG(hidden_dim)
        self.slca = CrossAttnAgg(hidden_dim)

        # cls
        self.head = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        clip_features, base_features, text_features = self.backbone(x)
        b, c, h, w = base_features.shape
        base_features = base_features.flatten(2).permute(0, 2, 1)

        clip_features = self.projection_cv(clip_features)
        text_features = self.projection_ct(text_features)

        base_features = self.projection_bv(base_features)

        scene_features = self.sfg(clip_features, base_features)
        scene_features = self.olca(scene_features.unsqueeze(1), base_features)

        class_features = self.cafg(text_features.unsqueeze(0).repeat(b, 1, 1), base_features)
        scene_features = self.slca(scene_features, class_features)

        # linear cls
        cls = self.head(scene_features).squeeze(1)

        class_features = self.projection_ct_ou(class_features)
        scene_features = self.projection_st_ou(scene_features)

        # class contrastive cls
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.repeat(b, 1, 1)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        similarity1 = 100 * (class_features * text_features).sum(-1)
        # scene contrastive cls
        scene_features = scene_features / scene_features.norm(dim=-1, keepdim=True)
        similarity2 = 100 * (scene_features @ text_features.permute(0, 2, 1)).squeeze(1)
        return cls, similarity1, similarity2


class CrossAttnAgg(nn.Module):

    def __init__(self, in_dim, head_count=8, ffn_ratio=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(in_dim, num_heads=head_count, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.mlp = MLP(in_dim, int(in_dim * ffn_ratio), in_dim, 2)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, target, image) -> torch.Tensor:
        target = self.norm1(target + self.dropout1(self.attn(target, image, image)[0]))
        target = self.norm2(target + self.dropout2(self.mlp(target)))
        return target


class CategoryAwareFG(nn.Module):

    def __init__(self, in_dim, head_count=8, ffn_ratio=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(in_dim, num_heads=head_count, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.mlp = MLP(in_dim, int(in_dim * ffn_ratio), in_dim, 2)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, target, image) -> torch.Tensor:
        target = self.norm1(self.dropout1(self.attn(target, image, image)[0]))
        target = self.norm2(target + self.dropout2(self.mlp(target)))
        return target


class SceneFG(nn.Module):

    def __init__(self, hidden_dim, ffn_ratio=4, dropout=0.1):
        super().__init__()

        self.mlp = MLP(hidden_dim, int(hidden_dim * ffn_ratio), hidden_dim, 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, clip, norm) -> torch.Tensor:
        scene = self.norm(self.dropout(self.mlp(clip + norm.mean(1))))
        return scene


class Backbone(nn.Module):
    def __init__(self, num_class=365, base_name='resnet50_places365', clip_name='ViT-B/32'):
        super(Backbone, self).__init__()
        
        # load clip model
        self.clip_backbone, _ = clip.load(clip_name, device='cpu')
        for p in self.clip_backbone.parameters():
            p.requires_grad = False

        if num_class == 365:
            classes = places365_classes
            print('load:', 'places365_classes')
        elif num_class == 397:
            classes = sun397_classes
            print('load:', 'sun397_classes')
        elif num_class == 67:
            classes = mit67_classes
            print('load:', 'mit67_classes')

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes])
        with torch.no_grad():
            self.text_features = self.clip_backbone.encode_text(text_inputs)
            
        # load base model
        base = base_name.split('_')[0]
        print('load:', base_name)
        if 'places365' in base_name:
            self.base_backbone = place_resnet.__dict__[base]()
            path = 'models/resnet50_places365.pth.tar'
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items() if
                          'fc' not in k}
            self.base_backbone.load_state_dict(state_dict, strict=False)
        elif 'pvt' in base_name:
            self.base_backbone = pvt_v2_b2()
            path = 'models/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.base_backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.base_backbone.load_state_dict(model_dict)
        else:
            self.base_backbone = place_resnet.__dict__[base](pretrained=True)

    def forward(self, x):

        with torch.no_grad():
            clip_image_features = self.clip_backbone.encode_image(x)
            text_features = self.text_features.to(x.device)
        base_image_features = self.base_backbone(x)

        return clip_image_features, base_image_features, text_features


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == '__main__':

    img = torch.rand((2, 3, 224, 224))

    model = OSFA()
    model.eval()
    out = model(img)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, (img))
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, (img))

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

    print(flop_count_table(flops, max_depth=2))

    # from thop import profile
    #
    # model = PSLNet(2048, 512, 12, 4, 1, 1)
    # model.eval()
    # flops, params = profile(model, inputs=(torch.ones(1, 3, 224, 224*4), ))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
