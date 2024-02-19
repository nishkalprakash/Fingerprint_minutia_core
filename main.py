from BI import Datasets,DS_PREFIX, Parallel
# from vars import DS_PREFIX
from pathlib import Path
import fingerprint_feature_extractor
import fingerprint_enhancer								# Load the library
import cv2
from tqdm import tqdm
def feature_vector(image_path, thres=10):
    
    # if image_path is of Path type then use image_path.as_posix()
    if isinstance(image_path, Path):
        image_path = image_path.as_posix()
    img = cv2.imread(image_path,0)
    img_enh = fingerprint_enhancer.enhance_Fingerprint(img)
    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img_enh, spuriousMinutiaeThresh=thres, invertImage=False, showResult=False, saveResult=False)
    # make a list of minutiae X Y Type Angle
    fv = []
    for m in FeaturesTerminations+FeaturesBifurcations:
        fv.append([int(m.locX), int(m.locY), m.Type == "Termination", float(m.Orientation[0])])
    doc = {
        'path':(Path(Path(image_path).parent.name)/Path(image_path).name).as_posix(), 
        f"mv_{thres}":fv
        }
    return doc

# def utkarsh_minutia(image_path):
#     # enhance the fingerprint image
#     enhanced_img = fingerprint_enhancer.enhance_Fingerprint(img)
#     # extract the minutiae features
#     FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(enhanced_img, spuriousMinutiaeThresh=20, invertImage=False, showResult=False, saveResult=True)
#     return FeaturesTerminations, FeaturesBifurcations

# def insert_into_db(path):
    # pass

if __name__ == "__main__":
        
    ds=Datasets("BI")
    p = Path("Anguli_10_100")
    cur=ds.connect(p.as_posix())
    base = DS_PREFIX/"anguli_fingerprint_datasets"/p

    images = list(base.glob("*.tiff"))
    # thres = 25
    ll = Parallel(debug=False)
    # to_process = ((i,thres) for i in images)
    # for i in images:
    for docs in ll(feature_vector,images,batch_size=1,chunksize=1):
        # push to mongodb cur
        # mv=feature_vector(i, thres)
        # cur.insert_one(doc)
        for doc in docs:
            path = doc.pop('path')
            cur.update_one({'path':path},{"$set":doc},upsert=True)
        # feature_vector(i)


    # img = cv2.imread('image_path', 0)						# read input image
    # out = fingerprint_enhancer.enhance_Fingerprint(img)		# enhance the fingerprint image
    # cv2.imshow('enhanced_image', out);						# display the result
    # cv2.waitKey(0)											# hold the display window

    # img = cv2.imread('image_path', 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
    # FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True)
    # # method